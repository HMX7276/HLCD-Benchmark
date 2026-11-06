# -*- coding: utf-8 -*-
"""Patient-grouped splits (default 5:1:3 via $SPLIT_RATIO; multi-admission
patients never cross splits)."""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

TL_DIR = config.TIMELINE_DIR
OUT = config.SPLITS_FILE


def build(seed: int | None = None):
    seed = seed if seed is not None else config.SPLITS_SEED
    parts = [float(x) for x in config.SPLIT_RATIO.split(":")]
    assert len(parts) == 3 and all(p >= 0 for p in parts), \
        f"SPLIT_RATIO must be train:val:test, got {config.SPLIT_RATIO}"
    aids = {}
    for p in sorted(glob.glob(os.path.join(TL_DIR, "*.json"))):
        doc = json.load(open(p, encoding="utf-8"))
        aids[doc["admission_id"]] = doc["patient_id"] or doc["admission_id"]
    patients = sorted({v for v in aids.values()})
    random.Random(seed).shuffle(patients)
    n = len(patients)
    tot = sum(parts)
    n_tr = round(n * parts[0] / tot)
    n_va = round(n * parts[1] / tot)
    groups = {"train": set(patients[:n_tr]),
              "val": set(patients[n_tr:n_tr + n_va]),
              "test": set(patients[n_tr + n_va:])}
    splits = {k: sorted(a for a, pid in aids.items() if pid in v)
              for k, v in groups.items()}
    return splits


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    splits = build()
    json.dump(splits, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in splits.items()}, "->", OUT)


if __name__ == "__main__":
    main()
