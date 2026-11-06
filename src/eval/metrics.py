# -*- coding: utf-8 -*-
"""Evaluation metrics (English / MIMIC edition):
B1 entity recall + time alignment (weak reference) + B3 multilabel macro-F1."""
import re
from src.eval.assertions import asserted_clauses, entity_text

B1_SCORE_VERSION = "b1-assertion-lexical-v2"


def norm(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    s = re.sub(r"\s+", " ", str(s or "")).strip().casefold()
    return s.rstrip(".,;:").strip()


_STOP = {"of", "due", "to", "with", "and", "the", "in", "on", "for", "a",
         "or", "by", "from", "unspecified", "other", "without", "type"}
_BRAND2GENERIC = {
    "lasix": "furosemide", "demadex": "torsemide", "aldactone": "spironolactone",
    "glucophage": "metformin", "coumadin": "warfarin", "synthroid": "levothyroxine",
    "norvasc": "amlodipine", "lipitor": "atorvastatin", "zocor": "simvastatin",
    "zoloft": "sertraline", "prozac": "fluoxetine", "coreg": "carvedilol",
    "lopressor": "metoprolol", "toprol": "metoprolol", "tenormin": "atenolol",
    "vasotec": "enalapril", "altace": "ramipril", "diovan": "valsartan",
    "cozaar": "losartan", "prilosec": "omeprazole", "protonix": "pantoprazole",
    "nexium": "esomeprazole", "xanax": "alprazolam", "ativan": "lorazepam",
    "valium": "diazepam", "dilantin": "phenytoin", "keflex": "cephalexin",
    "maxipime": "cefepime", "zosyn": "piperacillin tazobactam",
    "unasyn": "ampicillin sulbactam", "levaquin": "levofloxacin",
    "cipro": "ciprofloxacin", "flagyl": "metronidazole", "diflucan": "fluconazole",
}


def _tokens(s: str) -> set[str]:
    toks = set()
    for t in re.findall(r"[a-z0-9]+", norm(s)):
        t = _BRAND2GENERIC.get(t, t)
        if t not in _STOP and len(t) > 1:
            toks.add(t)
    return toks


def _lexical_match(ref: str, preds: list[str]) -> bool:
    r = norm(ref)
    if not r:
        return False
    rt = _tokens(ref)
    all_pt: set[str] = set()
    for p in preds:
        q = norm(p)
        if not q:
            continue
        if r == q:
            return True
        # 1) 词边界子串：短方 ≥5 字符且作为完整词组出现
        short, long_ = (q, r) if len(q) <= len(r) else (r, q)
        if len(short) >= 5 and re.search(rf"(?<!\w){re.escape(short)}(?!\w)", long_):
            return True
        # 2) 单条预测词元覆盖 ≥60%
        pt = _tokens(p)
        all_pt |= pt
        if rt and pt:
            cover = len(rt & pt) / len(rt)
            if cover >= 0.6 or (len(rt) <= 3 and rt <= pt):
                return True
    # 3) 全体预测并集覆盖 ≥60%（复合参考实体的分散召回）
    if rt and all_pt and len(rt & all_pt) / len(rt) >= 0.6:
        return True
    return False


def _match(ref: str, preds: list[str]) -> bool:
    # Scope lexical matching to compatible assertion clauses, never JSON keys.
    return any(ref_state == pred_state and _lexical_match(ref_clause, [pred_clause])
               for ref_state, ref_clause in asserted_clauses(ref)
               for pred in preds for pred_state, pred_clause in asserted_clauses(pred))


def b1_entity_metrics(reference: list[dict], prediction: dict) -> dict:
    by_type = {}
    for key in ("diagnoses", "surgeries", "medications", "allergy", "outcome"):
        v = prediction.get(key)
        if v is None:
            continue
        items = v if isinstance(v, list) else [v]
        by_type[key] = [entity_text(i) for i in items]
    type_map = {"discharge_diagnosis": "diagnoses",
                "procedure": "surgeries",
                "discharge_medication": "medications",
                "allergy": "allergy"}
    hits, total = 0, 0
    time_ok, time_total = 0, 0
    per_type = {}
    for e in reference:
        t = type_map.get(e["type"])
        if t is None:
            continue
        total += 1
        ok = _match(e["name"], by_type.get(t, []))
        hits += ok
        per_type.setdefault(e["type"], [0, 0])
        per_type[e["type"]][1] += 1
        per_type[e["type"]][0] += ok
        if ok and e["type"] == "procedure" and e.get("day") is not None:
            time_total += 1
            pred_days = []
            for i in prediction.get("surgeries", []):
                if isinstance(i, dict) and _match(e["name"], [entity_text(i)]):
                    d = i.get("day")
                    try:
                        pred_days.append(int(float(d)))
                    except (TypeError, ValueError):
                        pass
            if pred_days and any(abs(d - e["day"]) <= 1 for d in pred_days):
                time_ok += 1
    core_hits = sum(v[0] for t, v in per_type.items()
                    if t in ("discharge_diagnosis", "procedure"))
    core_total = sum(v[1] for t, v in per_type.items()
                     if t in ("discharge_diagnosis", "procedure"))
    return {"entity_recall": hits / total if total else None,
            "core_recall": core_hits / core_total if core_total else None,
            "time_alignment": time_ok / time_total if time_total else None,
            "n_ref": total,
            "per_type": {k: {"recall": v[0] / v[1], "n": v[1]}
                         for k, v in per_type.items()}}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def b3_multilabel_metrics(y_true: list[dict], y_pred: list[dict],
                          labels: list[str]) -> dict:
    if len(y_true) != len(y_pred) or not labels:
        raise ValueError("Aligned predictions and nonempty labels are required")
    if not y_true:
        return {"per_class": {}, "macro_f1": None, "exact_match": None, "n": 0}
    per = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t[lab] == 1 and p[lab] == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t[lab] == 0 and p[lab] == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t[lab] == 1 and p[lab] == 0)
        per[lab] = dict(zip(("precision", "recall", "f1"), _prf(tp, fp, fn)),
                        support=tp + fn)
    macro_f1 = sum(v["f1"] for v in per.values()) / len(per)
    exact = sum(1 for t, p in zip(y_true, y_pred)
                if all(t[k] == p.get(k, 0) for k in t)) / len(y_true)
    return {"per_class": per, "macro_f1": macro_f1, "exact_match": exact,
            "n": len(y_true)}
