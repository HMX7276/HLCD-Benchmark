# -*- coding: utf-8 -*-
"""End-to-end delivery verification. All thresholds derive from `.env`
(cohort size, traps per type, anchored-rate floor) -- no hardcoded counts."""
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

PY = sys.executable
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" -- {detail}" if detail else ""))


def n_rows(table: str) -> int:
    p = f"{config.MIMIC_RAW}/{table}.csv"
    if not os.path.exists(p):
        return -1
    with open(p, encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f) - 1


def main():
    n = config.COHORT_SIZE
    print("== 1. data ==")
    check("cohort.csv exists", os.path.exists(f"{config.MIMIC_RAW}/cohort.csv"))
    coh = pd.read_csv(f"{config.MIMIC_RAW}/cohort.csv") \
        if os.path.exists(f"{config.MIMIC_RAW}/cohort.csv") else pd.DataFrame()
    check(f"cohort == {n} (COHORT_SIZE)", len(coh) == n, str(len(coh)))
    for t, frac in [("discharge", 0.99), ("labevents", 150), ("prescriptions", 30),
                    ("radiology", 2), ("inputevents_blood", 0)]:
        rows = n_rows(t)
        lo = max(1, int(n * frac)) if t != "inputevents_blood" else 1
        check(f"table {t} >= {lo}", rows >= lo, f"{rows:,}")
    tls = glob.glob(os.path.join(config.TIMELINE_DIR, "*.json"))
    check(f"timelines == {n}", len(tls) == n, str(len(tls)))
    if tls:
        anchored = total = 0
        for p in tls:
            doc = json.load(open(p, encoding="utf-8"))
            total += len(doc["records"])
            anchored += sum(1 for r in doc["records"]
                            if r["granularity"] in ("exact", "day"))
        check(f"anchored >= {config.E2E_ANCHORED_MIN:.0%}",
              anchored / total >= config.E2E_ANCHORED_MIN,
              f"{anchored/total:.1%} of {total:,}")

    print("== 2. tasks ==")
    b1 = open(config.B1_DATASET, encoding="utf-8").readlines() \
        if os.path.exists(config.B1_DATASET) else []
    check(f"B1 >= {int(n * 0.99)}", len(b1) >= int(n * 0.99), str(len(b1)))
    if os.path.exists(config.SPLITS_FILE):
        sp = json.load(open(config.SPLITS_FILE, encoding="utf-8"))
        check("splits cover B1", sum(len(v) for v in sp.values()) >= len(b1) * 0.99,
              {k: len(v) for k, v in sp.items()})
    else:
        check("splits exist", False)
    from src.tasks.b3 import EVENTS, build_samples
    x_min = min(config.B3_X_VALUES)
    s1 = build_samples(x_min)
    check(f"B3 X={x_min} samples >= {int(n * 0.9)}", len(s1) >= int(n * 0.9),
          str(len(s1)))
    pos = {e: sum(1 for s in s1 if s["labels"][e]) for e in EVENTS}
    check("B3 all events have positives", all(v >= 1 for v in pos.values()), str(pos))
    traps = [json.loads(l) for l in open(config.TRAPS_FILE, encoding="utf-8")] \
        if os.path.exists(config.TRAPS_FILE) else []
    want = config.TRAPS_N_PER_TYPE * 6
    check(f"traps == {want} (TRAPS_N_PER_TYPE x 6)",
          len(traps) == want and len({t["type"] for t in traps}) == 6, str(len(traps)))

    print("== 3. tests ==")
    r = subprocess.run([PY, "-m", "unittest", "discover", "-s", "tests"],
                       capture_output=True, text=True, cwd=config.PROJECT_ROOT)
    check("unit tests pass", r.returncode == 0,
          r.stderr.strip().splitlines()[-1])

    print("== 4. experiments ==")
    for f, key, desc in [("b1_base.json", "core_recall_mean", "B1 baseline"),
                         ("b3_x1.json", "macro_f1", "B3 X=1"),
                         ("traps.json", None, "target-specific perturbations")]:
        p = os.path.join(config.RUNS_DIR, f)
        ok = os.path.exists(p)
        detail = ""
        if ok:
            m = json.load(open(p, encoding="utf-8"))
            mm = m.get("metrics", m)
            detail = str(mm.get(key)) if key else "ok"
        check(f"{desc} result", ok, detail)

    print("== 5. english-only code & config ==")
    cjk = re.compile(r"[\u4e00-\u9fff]")
    hits = []
    for path in glob.glob(os.path.join(config.PROJECT_ROOT, "src", "**", "*.py"),
                          recursive=True) + \
            glob.glob(os.path.join(config.PROJECT_ROOT, "tests", "*.py")) + \
            glob.glob(os.path.join(config.PROJECT_ROOT, "scripts", "*.py")):
        t = open(path, encoding="utf-8").read()
        t_no_comment = re.sub(r'#.*|"""[\s\S]*?"""', "", t)
        if cjk.search(t_no_comment):
            hits.append(os.path.relpath(path, config.PROJECT_ROOT))
    check("no CJK in code strings (comments excluded)", not hits, str(hits))
    check(".env exists (user config)", os.path.exists(
        os.path.join(config.PROJECT_ROOT, ".env")))
    check(".env.example tracked", os.path.exists(
        os.path.join(config.PROJECT_ROOT, ".env.example")))
    check("README exists", os.path.exists(
        os.path.join(config.PROJECT_ROOT, "README.md")))
    check("LLM provider configured", config.BASE_SLOT is not None,
          f"base slot: {config.BASE_SLOT[0] if config.BASE_SLOT else '-'}")

    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n==== e2e: {n_pass}/{len(CHECKS)} passed ====")
    os.makedirs(config.DOCS_DIR, exist_ok=True)
    json.dump([{"check": cn, "ok": ok, "detail": d} for cn, ok, d in CHECKS],
              open(os.path.join(config.DOCS_DIR, "e2e_checks.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
