# -*- coding: utf-8 -*-
"""B1 weak-reference dataset (MIMIC): entities parsed from the discharge
summary sections; the DS note itself is hidden from model input."""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.tasks.render import render_timeline

TL_DIR = config.TIMELINE_DIR
OUT = config.B1_DATASET

_NUM_SPLIT = re.compile(r"\n?\s*\d{1,2}[.:)]\s+")
_UNIT_TOK = re.compile(r"\d")
_NON_DRUG = {"mg", "mcg", "g", "gm", "ml", "tab", "tabs", "tablet", "tablets",
             "cap", "caps", "capsule", "po", "iv", "sc", "im", "prn", "daily",
             "qd", "bid", "tid", "qhs", "qam", "q6h", "q8h", "q12h", "x", "as",
             "needed", "puff", "puffs", "unit", "units", "apply", "take",
             "inhale", "sig", "sig:", "each", "every", "disp", "dispense",
             "tablet", "capsule", "times", "day", "days", "night", "route"}


_MED_ITEM = re.compile(r"(?:^|\n)\s*\d{1,2}[.)]\s+")


def _med_names(text: str) -> list[str]:
    """'1. Acetaminophen 325-650 mg PO ...' -> ['Acetaminophen', ...]；
    续行（RX/Disp/说明）只归并进条目、不参与药名抽取。"""
    items = _MED_ITEM.split(text)[1:]
    if not items:  # 无编号版式：按行兜底
        items = [l for l in text.splitlines() if l.strip()]
    out = []
    for item in items:
        first_line = item.strip().splitlines()[0] if item.strip() else ""
        first_line = re.sub(r"\([^)]*\)", "", first_line)
        words = []
        for tok in first_line.split():
            if _UNIT_TOK.search(tok) or tok.lower().strip("*.,:") in _NON_DRUG:
                break
            words.append(tok.strip("*.,:"))
        name = " ".join(words).strip()
        if 3 <= len(name) <= 60 and not name.lower().startswith("reason"):
            out.append(name)
    return out


def _dx_list(text: str) -> list[str]:
    items = []
    for part in _NUM_SPLIT.split("\n" + text.strip()):
        part = part.strip().rstrip(".,;")
        part = re.sub(r"^(?:primary|secondary|tertiary)\s*:\s*", "", part,
                      flags=re.I)
        # 无编号的复合行：按逗号/分号再切
        for sub in re.split(r"[;,]\s*(?=[A-Za-z])", part):
            sub = sub.strip().rstrip(".,;")
            if 3 <= len(sub) <= 200:
                items.append(re.sub(r"\s+", " ", sub))
    return items


def _proc_list(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line.strip().rstrip(".,;"))
        if 4 <= len(line) <= 150 and not line.startswith("("):
            out.append(line)
    return out


def build_one(doc: dict) -> dict | None:
    ds = next((r for r in doc["records"] if r["kind"] == "note_ds"), None)
    if not ds:
        return None
    sec = ds["content"]["sections"]
    entities = []
    for x in _dx_list(sec.get("discharge diagnosis", "")):
        entities.append({"type": "discharge_diagnosis", "name": x})
    for x in _proc_list(sec.get("major surgical or invasive procedure", "")):
        entities.append({"type": "procedure", "name": x})
    # 结构化手术编码参考（有 day，供时间对齐；DS 文本日期已被 ___ 掩码）
    for r in doc["records"]:
        if r["kind"] == "procedure" and r["day"] is not None:
            entities.append({"type": "procedure", "name": r["content"]["name"],
                             "day": r["day"]})
    for x in _med_names(sec.get("discharge medication", "")):
        entities.append({"type": "discharge_medication", "name": x})
    # 注：过敏信息仅存在于（对模型隐藏的）出院小结文本中，MIMIC 结构化表无
    # 过敏字段，属结构性不可推断，不作为参考实体（避免召回上限<1 的假指标）。
    if not entities:
        return None
    return {
        "admission_id": doc["admission_id"],
        "patient_id": doc["patient_id"],
        "hadm_id": doc["meta"]["hadm_id"],
        "input": render_timeline(doc, hide_discharge=True),
        "entities": entities,
        "reference_summary": sec.get("hospital course", "")[:4000],
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for p in sorted(glob.glob(os.path.join(TL_DIR, "*.json"))):
            doc = json.load(open(p, encoding="utf-8"))
            item = build_one(doc)
            if item:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
    print(f"B1 dataset: {n} admissions -> {OUT}")


if __name__ == "__main__":
    main()
