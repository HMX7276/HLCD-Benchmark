"""Single-model B3 runner with strict binary response validation."""
import argparse
import glob
import json
import os
from src import config
from src.runners import prompts
from src.runners.common import (infer, binary_prediction, model_options, run_fingerprint,
                                execute, save_result)
from src.eval.metrics import b3_multilabel_metrics
from src.tasks.b3 import EVENTS, build_samples, B3_LABEL_VERSION
from src.tasks.render import render_timeline
from src.tasks.temporal import B3_INPUT_POLICY
from src.tasks.b3_provenance import validate_timeline
TL_DIR = config.TIMELINE_DIR

def load_samples(n, X, seed=None, split="test", audit=None):
    import random
    seed = seed if seed is not None else config.EVAL_SEED
    label_audit = {}
    ss = build_samples(X, audit=label_audit)
    print("[B3 label eligibility] " + json.dumps(label_audit, sort_keys=True))
    if split:
        # 与 B1 同口径：先在指定 split（默认 test）内过滤，再按种子抽样——
        # 旧版在全队列抽样，约 2/3 样本落在 train/val，与 B1 子集不可比
        with open(config.SPLITS_FILE, encoding="utf-8") as stream:
            splits = json.load(stream)
        keep = set(splits[split])
        ss = [s for s in ss if s["admission_id"] in keep]
    if n and n < len(ss):
        ss = random.Random(seed).sample(ss, n)
    if audit is not None:
        audit.update(label_audit)
        audit['selected_after_split_and_sampling'] = len(ss)
    need = {s["admission_id"] for s in ss}
    tls = {}
    for p in glob.glob(os.path.join(TL_DIR, "*.json")):
        with open(p, encoding="utf-8") as stream:
            doc = json.load(stream)
        if doc["admission_id"] in need:
            tls[doc["admission_id"]] = doc
    return ss, tls

def validate(parsed):
    return binary_prediction(parsed, EVENTS)


def run_baseline(model, timeline, X, endpoint=None, url_override=None, key_override=None):
    messages = [{"role":"system", "content":prompts.B3_SYSTEM.format(X=X, X1=X+1)},
                {"role":"user", "content":prompts.B3_USER.format(timeline=timeline,X=X,X1=X+1)}]
    return infer(messages, validate, model, endpoint, url_override, key_override)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--endpoint')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--X', type=int, default=3)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default='')
    ap.add_argument('--ckpt', default='')
    args = ap.parse_args()
    if args.n < 0 or args.X < 0: ap.error('--n and --X must be nonnegative')
    eligibility_audit = {}
    samples, timelines = load_samples(args.n,args.X,split=args.split or None,audit=eligibility_audit)
    options = model_options(args)
    fingerprint = run_fingerprint('B3', options, {'X':args.X,'labels':B3_LABEL_VERSION,'input':B3_INPUT_POLICY})
    items, audits = [], {}
    for sample in samples:
        aid = sample['admission_id']; doc = timelines[aid]
        validate_timeline(doc)
        audit = {}
        text = render_timeline(doc,max_day=args.X,audit=audit)
        audits[aid] = audit
        items.append({'id':aid, 'input':text, 'context':{'X':args.X,'labels':sample['labels']}})
    def work(item):
        return run_baseline(options[0],item['input'],args.X,options[1],options[2],options[3])
    rows = execute(items,work,validate,'B3',fingerprint,args.ckpt)
    labels = {s['admission_id']:{e:s['labels'][e] for e in EVENTS} for s in samples}
    good = [row for row in rows if row['status']=='ok']
    metrics = b3_multilabel_metrics([labels[row['id']] for row in good], [row['prediction'] for row in good], EVENTS)
    metrics.update(n_total=len(rows),n_valid=len(good),fails=len(rows)-len(good),model=options[0],
                   X=args.X,label_version=B3_LABEL_VERSION,input_policy=B3_INPUT_POLICY,
                   run_fingerprint=fingerprint,eligibility_audit=eligibility_audit,
                   denominator='valid responses only; failures excluded and counted')
    for row in rows:
        row['y_true'] = labels[row['id']]
        row['input_audit'] = audits[row['id']]
    save_result(args.out,metrics,rows)

if __name__ == '__main__': main()
