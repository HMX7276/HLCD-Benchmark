# -*- coding: utf-8 -*-
"""Heuristic (LLM-free) extractive baseline for the B1 track.

Competes in B1 without any model call: entities are lifted verbatim from the
rendered timeline using the same conventions the renderer itself emits
(``=== Day N ===`` separators, ``[Lab]`` / ``[Medication]`` /
``[Procedure]`` / ``[Radiology]`` prefixes -- keep in sync with
src/tasks/render.py).

Two entities are structurally out of reach and returned empty (documented,
not a bug): diagnoses exist only in the hidden discharge summary and the
hidden coded-diagnosis records, and allergy information only exists in the
discharge summary. The recall gap between this baseline and LLMs therefore
quantifies how much of B1 requires inference beyond verbatim extraction
(a deterministic lexical baseline).

Run:
  python -m src.baselines.extractive_b1 --n 30 \
      --out $HLCD_WORK/mimic_runs/b1_base_extractive.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.runners.run_b1 import evaluate, load_cases
from src.eval.metrics import norm
from src.tasks.b1 import _NON_DRUG, _UNIT_TOK  # single source of truth

_DAY_RE = re.compile(r"^=== Day (-?\d+) ===\s*$", re.M)
_ROW_RE = re.compile(r"^\[(\w+)\]\s?(.*)$")
_LAB_RE = re.compile(r"^(?P<name>.+?)\s=\s(?P<rest>.+)$")
_FLAG_RE = re.compile(r"\[(abnormal|critical|hh|ll|h|l|aa)\]")
_DIRECTION = {"h": "high", "hh": "high", "l": "low", "ll": "low",
              "aa": "abnormal", "abnormal": "abnormal",
              "critical": "critical"}


def parse_timeline(text: str) -> list[dict]:
    """Rendered timeline -> [{day, kind, content}, ...] in document order."""
    rows = []
    marks = list(_DAY_RE.finditer(text))
    for i, m in enumerate(marks):
        day = int(m.group(1))
        seg_end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        for line in text[m.end():seg_end].splitlines():
            r = _ROW_RE.match(line.strip())
            if r:
                rows.append({"day": day, "kind": r.group(1),
                             "content": r.group(2).strip()})
    return rows


def _med_name(text: str) -> str | None:
    """Leading drug-name tokens: stop at the first dose/unit/route token."""
    text = re.sub(r"\([^)]*\)", "", text)  # drop parenthesised dose notes
    words = []
    for tok in text.split():
        if _UNIT_TOK.search(tok) or tok.lower().strip("*.,:") in _NON_DRUG:
            break
        words.append(tok.strip("*.,:"))
    name = " ".join(words).strip()
    return name if 3 <= len(name) <= 60 else None


def _extract_medications(rows) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        names: list[str] = []
        if r["kind"] == "Medication":
            n = _med_name(r["content"])
            if n:
                names.append(n)
        elif r["kind"] == "MedAdmin" and r["content"].startswith("administered:"):
            for part in r["content"][len("administered:"):].split(","):
                p = part.strip()
                if len(p) >= 3:
                    names.append(p)
        for n in names:
            k = norm(n)
            if k and k not in seen:
                seen.add(k)
                out.append({"name": n, "day": r["day"]})
    return out


def _extract_surgeries(rows) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        if r["kind"] not in ("Procedure", "ICUProcedure"):
            continue
        name = re.split(r"\s=\s", r["content"], maxsplit=1)[0].strip()
        k = norm(name)
        if name and k not in seen:
            seen.add(k)
            out.append({"name": name, "day": r["day"]})
    return out


def _extract_abnormal_labs(rows) -> list[dict]:
    out = []
    for r in rows:
        if r["kind"] != "Lab":
            continue
        m = _LAB_RE.match(r["content"])
        if not m:
            continue
        f = _FLAG_RE.search(m.group("rest"))
        if not f:  # normal result: folded by the renderer, not a key lab
            continue
        direction = _DIRECTION[f.group(1)]
        value_part = m.group("rest")[:f.start()].strip()
        unit = m.group("rest")[f.end():].strip()
        value = f"{value_part} {unit} ({direction})".strip()
        out.append({"name": m.group("name").strip(), "value": value,
                    "day": r["day"]})
    return out


def _extract_imaging(rows) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        if r["kind"] != "Radiology":
            continue
        exam, _, impression = r["content"].partition(": ")
        name = (impression or exam).strip()
        k = norm(name)[:80]
        if name and k not in seen:
            seen.add(k)
            out.append({"name": name, "day": r["day"]})
    return out


def build_prediction(text: str) -> dict:
    rows = parse_timeline(text)
    max_day = max((r["day"] for r in rows), default=0)
    discharged = any(r["kind"] == "Transfer"
                     and "discharge" in r["content"].lower() for r in rows)
    outcome = (f"The patient was discharged on Day {max_day}." if discharged
               else f"The timeline extends to Day {max_day}.")
    return {"diagnoses": [],  # hidden with the discharge summary: unreachable
            "surgeries": _extract_surgeries(rows),
            "medications": _extract_medications(rows),
            "key_abnormal_labs": _extract_abnormal_labs(rows),
            "key_imaging_findings": _extract_imaging(rows),
            "allergy": "",  # only documented in the hidden summary
            "outcome": outcome}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="subsample size (0 = full split)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = load_cases(args.n or None, args.split)
    results = [(c, build_prediction(c["input"])) for c in cases]
    metrics = evaluate(results)
    metrics.update({"mode": "base", "model": "extractive-heuristic",
                    "fails": 0})
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        json.dump({"metrics": metrics,
                   "predictions": [{"admission_id": c["admission_id"],
                                    "pred": p} for c, p in results]},
                  open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
