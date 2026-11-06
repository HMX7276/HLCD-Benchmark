# -*- coding: utf-8 -*-
"""Generate a timestamped run report: reports/{YYYYMMDD_HHMMSS}.md.

Aggregates config snapshot (secrets masked), cohort profile, timeline stats,
task dataset stats, experiment results, and the latest e2e checks."""
import glob
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def _mask(v):
    return "***" if v else ""


def config_snapshot():
    slot = config.BASE_SLOT
    return [
        ("LLM", f"{slot[0] if slot else '-'}"
                f"{' (key set)' if (slot and slot[3]) else ' (no key)'}"),
        ("LLM_THINKING", config.LLM_THINKING),
        ("PostgreSQL", f"{config.PG['host']}:{config.PG['port']} "
                       f"{config.PG_DB_MIMICIV}+{config.PG_DB_MIMICNOTE} "
                       f"(pwd {_mask(config.PG['password'])})"),
        ("COHORT_SIZE / SEED / MIN_LOS", f"{config.COHORT_SIZE} / {config.COHORT_SEED}"
                                         f" / {config.MIN_LOS_DAYS}"),
        ("TIMELINE_VERSION", config.TIMELINE_VERSION),
        ("SPLIT_RATIO / seed", f"{config.SPLIT_RATIO} / {config.SPLITS_SEED}"),
        ("B3_X_VALUES / NEW_EXAM_LABS", f"{config.B3_X_VALUES} / {config.B3_NEW_EXAM_LABS}"),
        ("TRAPS_N_PER_TYPE / seed", f"{config.TRAPS_N_PER_TYPE} / {config.TRAPS_SEED}"),
        ("LLM limits", f"max_tokens {config.LLM_MAX_TOKENS}, retries {config.LLM_RETRIES}"
                       f", timeout {config.LLM_TIMEOUT}s, budget cap {config.LLM_MAX_TOKEN_BUDGET}"),
    ]


def cohort_stats():
    import pandas as pd
    p = f"{config.MIMIC_RAW}/cohort.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df


def timeline_stats():
    files = sorted(glob.glob(os.path.join(config.TIMELINE_DIR, "*.json")))
    if not files:
        return None
    n_rec = anchored = 0
    kinds = Counter()
    for fp in files:
        for r in json.load(open(fp, encoding="utf-8"))["records"]:
            n_rec += 1
            kinds[r["kind"]] += 1
            anchored += r["granularity"] in ("exact", "day")
    return {"n": len(files), "records": n_rec, "anchored": anchored, "kinds": kinds}


def b1_stats():
    if not os.path.exists(config.B1_DATASET):
        return None
    rows = [json.loads(l) for l in open(config.B1_DATASET, encoding="utf-8")]
    per_type = Counter(e["type"] for r in rows for e in r["entities"])
    chars = sorted(len(r["input"]) for r in rows)
    return {"n": len(rows), "mean_entities": sum(per_type.values()) / max(1, len(rows)),
            "per_type": per_type, "input_chars_median": chars[len(chars) // 2],
            "input_chars_p95": chars[int(len(chars) * .95)]}


def b3_stats():
    from src.tasks.b3 import EVENTS, build_samples
    out = {}
    for X in config.B3_X_VALUES:
        ss = build_samples(X)
        if not ss:
            continue
        pos = {e: sum(1 for s in ss if s["labels"][e]) for e in EVENTS}
        out[X] = {"samples": len(ss), "positives": pos}
    return out


def traps_stats():
    if not os.path.exists(config.TRAPS_FILE):
        return None
    return Counter(json.loads(l)["type"]
                   for l in open(config.TRAPS_FILE, encoding="utf-8"))


def experiments():
    out = []
    for f, desc in [("b1_base.json", "B1 single-model baseline"),
                    ("b3_x1.json", "B3 X=1"),
                    ("traps.json", "Hallucination traps")]:
        p = os.path.join(config.RUNS_DIR, f)
        if not os.path.exists(p):
            continue
        m = json.load(open(p, encoding="utf-8"))
        mm = m.get("metrics", m)
        if "entity_recall_mean" in mm:
            detail = (f"recall {mm['entity_recall_mean']}, core {mm.get('core_recall_mean')}"
                      f", per-type {mm.get('per_type')}")
        elif "macro_f1" in mm:
            detail = (f"macro-F1 {mm['macro_f1']}, exact {mm.get('exact_match')}")
        else:
            detail = json.dumps(mm.get("per_type", {}), ensure_ascii=False)
        out.append((desc, mm.get("model", "-"), mm.get("n", mm.get("fails", "")), detail))
    return out


def e2e_summary():
    p = os.path.join(config.DOCS_DIR, "e2e_checks.json")
    if not os.path.exists(p):
        return None
    checks = json.load(open(p, encoding="utf-8"))
    n_pass = sum(1 for c in checks if c["ok"])
    fails = [c["check"] for c in checks if not c["ok"]]
    return {"total": len(checks), "passed": n_pass, "fails": fails}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    git = ""
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=config.PROJECT_ROOT).stdout.strip()
    except Exception:
        pass

    L = []
    L.append(f"# Benchmark Run Report — {ts}\n")
    L.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- git: `{git or 'n/a'}`\n")

    L.append("## Configuration (`.env` snapshot, secrets masked)\n")
    L.append("| key | value |")
    L.append("|---|---|")
    for k, v in config_snapshot():
        L.append(f"| {k} | {v} |")
    L.append("")

    coh = cohort_stats()
    if coh is not None:
        L.append("## Cohort\n")
        L.append(f"- admissions: **{len(coh)}** (COHORT_SIZE={config.COHORT_SIZE})")
        L.append(f"- ICU: {coh.has_icu.mean():.0%} | expired: "
                 f"{coh.hospital_expire_flag.mean():.0%} | "
                 f"LOS median {coh.los_days.median():.1f}d "
                 f"(P25 {coh.los_days.quantile(.25):.1f} / P75 {coh.los_days.quantile(.75):.1f})")
        at = coh.admission_type.value_counts()
        L.append("- admission types: " + ", ".join(f"{k} {v}" for k, v in at.items()))
        L.append("")

    ts_ = timeline_stats()
    if ts_:
        L.append("## Timelines\n")
        L.append(f"- admissions: **{ts_['n']}** | records: **{ts_['records']:,}** | "
                 f"anchored (exact/day): **{ts_['anchored']/ts_['records']:.1%}**")
        L.append("- records by kind: " + ", ".join(
            f"{k} {v:,}" for k, v in ts_["kinds"].most_common()))
        L.append("")

    b1 = b1_stats()
    if b1:
        L.append("## B1 dataset (weak reference)\n")
        L.append(f"- cases: **{b1['n']}** | mean reference entities: "
                 f"{b1['mean_entities']:.1f}")
        L.append("- entities by type: " + ", ".join(
            f"{k} {v:,}" for k, v in b1["per_type"].most_common()))
        L.append(f"- input length: median {b1['input_chars_median']:,} chars, "
                 f"P95 {b1['input_chars_p95']:,}")
        L.append("")

    b3 = b3_stats()
    if b3:
        L.append("## B3 label distribution\n")
        for X, d in b3.items():
            pos = ", ".join(f"{k} {v}" for k, v in
                            sorted(d["positives"].items(), key=lambda kv: -kv[1]))
            L.append(f"- X={X}: {d['samples']:,} samples; positives — {pos}")
        L.append("")

    tr = traps_stats()
    if tr:
        L.append("## Hallucination traps\n")
        L.append("- " + ", ".join(f"{k} {v}" for k, v in sorted(tr.items()))
                 + f" (total {sum(tr.values())})\n")

    ex = experiments()
    if ex:
        L.append("## Experiment results (this run)\n")
        L.append("| experiment | model | n | metrics |")
        L.append("|---|---|---|---|")
        for desc, model, n, detail in ex:
            L.append(f"| {desc} | {model} | {n} | {detail} |")
        L.append("")

    e2e = e2e_summary()
    if e2e:
        L.append("## e2e verification\n")
        L.append(f"- **{e2e['passed']}/{e2e['total']}** checks passed"
                 + (f"; failures: {e2e['fails']}" if e2e["fails"] else " — all green"))
        L.append("")

    out = os.path.join(config.REPORTS_DIR, f"{ts}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print(f"report -> {out}")
    return out


if __name__ == "__main__":
    main()
