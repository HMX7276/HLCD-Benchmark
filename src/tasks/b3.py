# -*- coding: utf-8 -*-
"""B3 event derivation from MIMIC timelines (structured fields only).
Label set is finalized from measured distribution (per task spec clause)."""
import glob
import json
import os
import sys
import warnings
from datetime import timedelta
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.tasks.temporal import timestamp

TL_DIR = config.TIMELINE_DIR
B3_LABEL_VERSION = "b3-exact-events-v2"


def event_day(at, admit):
    event, start = timestamp(at), timestamp(admit)
    if event is None or start is None or event < start:
        raise ValueError("Missing or invalid exact admission/event timestamp")
    return (event - start).days

# 定稿事件集（单一事实源；键名与 prompts.B3_SYSTEM 的 JSON 键同步）
EVENTS = ["discharge", "death", "icu_transfer", "antibiotic_start",
          "antibiotic_stop", "extubation", "transfusion", "major_procedure",
          "new_key_exam"]

_ABX = ("vancomycin", "ceftriaxone", "cefepime", "cefazolin", "ceftazidime",
        "ertapenem", "piperacillin", "tazobactam", "meropenem", "imipenem",
        "ciprofloxacin", "levofloxacin", "moxifloxacin", "metronidazole",
        "azithromycin", "ampicillin", "amoxicillin", "nafcillin", "oxacillin",
        "clindamycin", "doxycycline", "gentamicin", "tobramycin", "amikacin",
        "linezolid", "daptomycin", "fluconazole", "micafungin", "caspofungin",
        "sulfamethoxazole", "trimethoprim", "nitrofurantoin", "cefuroxime",
        "penicillin")
_BLOOD = ("packed red blood", "pooled platelet", "platelet", "fresh frozen",
          "cryoprecipitate", "red blood cells", "prbc", "ffp", "whole blood")


def derive_events(doc: dict) -> dict[str, list[int]]:
    ev = defaultdict(list)
    m = doc["meta"]
    ev["discharge"].append(event_day(m.get("discharge_time"), m.get("admit_time")))
    if m.get("death"):
        # Unknown death times are not silently replaced by discharge times.
        death_day = event_day(m.get("death_time"), m.get("admit_time"))
        if timestamp(m['death_time']) > timestamp(m['discharge_time']):
            raise ValueError("Death time follows discharge for an in-hospital death label")
        ev["death"].append(death_day)
    abx_seen = False
    for r in doc["records"]:
        c, k = r["content"], r["kind"]
        day = r["day"]
        if k == "transfer" and day is not None:
            cu = str(c.get("careunit") or "")
            if "ICU" in cu and c.get("eventtype") in ("transfer", "admit", "ICU"):
                ev["icu_transfer"].append(day)
        elif k == "med" and day is not None:
            drug = str(c.get("drug") or "").lower()
            if any(a in drug for a in _ABX):
                if not abx_seen:
                    ev["antibiotic_start"].append(day)
                    abx_seen = True
                if c.get("stop_day") is not None:
                    ev["antibiotic_stop"].append(int(c["stop_day"]))
        elif k == "transfusion" and day is not None:
            ev["transfusion"].append(day)
        elif k == "icu_procedure" and day is not None:
            if "extuba" in str(c.get("name") or "").lower():
                ev["extubation"].append(day)
        elif k == "procedure" and day is not None:
            ev["major_procedure"].append(day)
    # new_key_exam：单日检验 itemid 去重 ≥5 或有影像
    per_day = defaultdict(lambda: {"lab_items": set(), "img": 0})
    for r in doc["records"]:
        if r["day"] is None:
            continue
        if r["kind"] == "lab":
            per_day[r["day"]]["lab_items"].add(str(r["source"].get("itemid")))
        elif r["kind"] == "radiology":
            per_day[r["day"]]["img"] += 1
    for d, v in per_day.items():
        if len(v["lab_items"]) >= config.B3_NEW_EXAM_LABS or v["img"] >= 1:
            ev["new_key_exam"].append(d)
    return {e: sorted(set(v)) for e, v in ev.items()}


def build_samples(X: int, events: list[str] | None = None, audit=None) -> list[dict]:
    if type(X) is not int or X < 0:
        raise ValueError("X must be a nonnegative integer")
    events = events or EVENTS
    out = []
    counts = Counter()
    for p in sorted(glob.glob(os.path.join(TL_DIR, "*.json"))):
        with open(p, encoding="utf-8") as stream:
            doc = json.load(stream)
        counts['total'] += 1
        admit = timestamp(doc['meta'].get('admit_time'))
        discharge = timestamp(doc['meta'].get('discharge_time'))
        if admit is None or discharge is None or discharge < admit:
            counts['excluded_invalid_admission_or_discharge'] += 1
            continue
        cutoff = admit + timedelta(days=X+1)
        if discharge < cutoff:
            counts['excluded_discharged_before_cutoff'] += 1
            continue
        try:
            ev = derive_events(doc)
        except ValueError:
            counts['excluded_invalid_or_missing_death_time'] += 1
            continue
        if doc['meta'].get('death') and timestamp(doc['meta']['death_time']) < cutoff:
            counts['excluded_died_before_cutoff'] += 1
            continue
        labels = {e: int(X + 1 in ev.get(e, [])) for e in events}
        labels["no_event"] = int(all(v == 0 for k, v in labels.items()))
        out.append({"admission_id": doc["admission_id"],
                    "hadm_id": doc["meta"]["hadm_id"], "X": X,
                    "day_target": X + 1, "labels": labels})
    counts['eligible'] = len(out)
    if audit is not None:
        audit.update(counts)
    elif any('invalid' in key for key in counts):
        warnings.warn(f"B3 label exclusions: {dict(counts)}", RuntimeWarning)
    return out


def main():
    for X in config.B3_X_VALUES:
        audit = {}
        ss = build_samples(X, audit=audit)
        print(f"X={X} eligibility: {audit}")
        if not ss:
            continue
        pos = Counter()
        for s in ss:
            for e, v in s["labels"].items():
                if v and e != "no_event":
                    pos[e] += 1
        print(f"\n== X={X}: samples {len(ss)}, positives ==")
        for e, n in pos.most_common():
            print(f"  {e}: {n} ({n/len(ss):.1%})")


if __name__ == "__main__":
    main()
