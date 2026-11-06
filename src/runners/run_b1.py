"""Single-model B1 evaluation; failures remain separate from valid responses."""
import argparse
import json
import os
from src import config
from src.runners import prompts
from src.runners.common import (infer, summary_prediction, model_options, run_fingerprint,
                                execute, save_result)
from src.eval.metrics import b1_entity_metrics, B1_SCORE_VERSION
DS = config.B1_DATASET

def load_cases(n=None, split="test", seed=None):
    seed = seed if seed is not None else config.EVAL_SEED
    with open(DS, encoding="utf-8") as stream:
        rows = [json.loads(l) for l in stream if l.strip()]
    if split:
        with open(config.SPLITS_FILE, encoding="utf-8") as stream:
            splits = json.load(stream)
        rows = [r for r in rows
                if r["admission_id"] in set(splits[split])]
    if n and n < len(rows):
        import random
        rows = random.Random(seed).sample(rows, n)
    return rows

def evaluate(results: list) -> dict:
    recs = [b1_entity_metrics(c["entities"], p) for c, p in results]
    ok = [r for r in recs if r["entity_recall"] is not None]
    import statistics as st
    if not ok:  # all cases failed (e.g. endpoint down): report honestly
        return {"n": 0, "entity_recall_mean": None, "core_recall_mean": None,
                "time_alignment_mean": None, "per_type": {}}
    mean_recall = st.mean(r["entity_recall"] for r in ok)
    core = [r["core_recall"] for r in ok if r["core_recall"] is not None]
    ta = [r["time_alignment"] for r in ok if r["time_alignment"] is not None]
    return {"n": len(ok), "entity_recall_mean": round(mean_recall, 4),
            "core_recall_mean": (round(st.mean(core), 4) if core else None),
            "time_alignment_mean": (round(st.mean(ta), 4) if ta else None),
            "per_type": _agg_per_type(ok)}

def _agg_per_type(recs):
    agg = {}
    for r in recs:
        for t, v in r["per_type"].items():
            a = agg.setdefault(t, [0.0, 0])
            a[0] += v["recall"] * v["n"]
            a[1] += v["n"]
    return {t: round(h / n, 4) for t, (h, n) in agg.items()}

def run_baseline(model, case, endpoint=None, url_override=None, key_override=None):
    messages = [{"role":"system", "content":prompts.B1_SYSTEM},
                {"role":"user", "content":prompts.B1_USER.format(timeline=case["input"])}]
    return infer(messages, summary_prediction, model, endpoint, url_override, key_override)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--endpoint')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default='')
    ap.add_argument('--ckpt', default='')
    args = ap.parse_args()
    if args.n < 0: ap.error('--n must be nonnegative')
    cases = load_cases(args.n, args.split or None)
    options = model_options(args)
    fingerprint = run_fingerprint('B1', options, B1_SCORE_VERSION)
    items = [{'id':c['admission_id'], 'input':c['input'], 'context':c['entities']} for c in cases]
    def work(item):
        return run_baseline(options[0], item, options[1], options[2], options[3])
    rows = execute(items, work, summary_prediction, 'B1', fingerprint, args.ckpt)
    by_id = {c['admission_id']:c for c in cases}
    metrics = evaluate([(by_id[row['id']], row['prediction']) for row in rows if row['status']=='ok'])
    valid = sum(row['status']=='ok' for row in rows)
    metrics.update(n_total=len(rows), n_valid=valid, fails=len(rows)-valid, model=options[0],
                   score_version=B1_SCORE_VERSION, run_fingerprint=fingerprint,
                   denominator='valid responses with nonempty reference')
    save_result(args.out, metrics, rows)

if __name__ == '__main__': main()
