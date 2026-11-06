# -*- coding: utf-8 -*-
"""MIMIC timeline -> compact English text (B1/B3 model input).

Coupling contracts (keep in sync atomically):
- "=== Day N ===" separator  <- relative-day input grouping
- "[Lab]"/"[Med]" prefixes   <- traps/build
"""
from collections import defaultdict

from src import config
from src.tasks.temporal import RESULT_KINDS, prediction_cutoff, result_visible

_KIND_HEAD = {"lab": "Lab", "med": "Medication", "emar": "MedAdmin",
              "procedure": "Procedure", "icu_procedure": "ICUProcedure",
              "transfer": "Transfer", "radiology": "Radiology",
              "vitals": "Vitals", "gcs": "GCS", "microbio": "Microbiology",
              "diagnosis": "Diagnosis", "note_ds": "DischargeNote",
              "admission": "Admission", "transfusion": "Transfusion"}

_ABNORMAL = {"abnormal", "critical", "h", "hh", "l", "ll", "aa"}


def _fmt_lab(r) -> str:
    c = r["content"]
    flag = (c.get("flag") or "").lower()
    mark = f" [{flag}]" if flag in _ABNORMAL else ""
    unit = f" {c['unit']}" if c.get("unit") else ""
    return f"{c['name']} = {c['value']}{mark}{unit}"


def _fmt(r) -> str:
    c, k = r["content"], r["kind"]
    if k == "lab":
        return _fmt_lab(r)
    if k == "med":
        stop = f" (stop Day {c['stop_day']})" if c.get("stop_day") is not None else ""
        dose = f" {c['dose']}" if c.get("dose") else ""
        route = f" {c['route']}" if c.get("route") else ""
        return f"{c['drug']}{dose}{route}{stop}"
    if k == "emar":
        return str(c.get("medication") or "")
    if k == "procedure":
        return f"{c.get('name', '')}"
    if k == "icu_procedure":
        v = f" = {c['value']}" if c.get("value") else ""
        return f"{c.get('name', '')}{v}"
    if k == "transfer":
        cu = c.get("careunit")
        if c.get("eventtype") == "discharge":
            return "ward discharge/disposition event"
        return f"{c.get('eventtype')} -> {cu if cu and cu != 'nan' else 'ward'}"
    if k == "radiology":
        imp = (c.get("impression") or "").replace("\n", " ")[:200]
        return f"{(c.get('exam') or 'imaging').strip()[:60]}: {imp}"
    if k == "vitals":
        c = {k2: round(v, 1) for k2, v in c.items()}
        return " ".join(f"{k2}={v}" for k2, v in
                        [("HR", c.get("heart_rate")), ("SBP", c.get("sbp")),
                         ("DBP", c.get("dbp")), ("RR", c.get("resp_rate")),
                         ("T", c.get("temperature")),
                         ("SpO2", c.get("o2sat"))] if v is not None)
    if k == "gcs":
        return f"min GCS {c.get('min_gcs'):.0f}"
    if k == "microbio":
        org = c.get("org") or ""
        interp = f" [{c['interpretation']}]" if c.get("interpretation") else ""
        return f"{c.get('test', '')}: {org}{interp}"
    if k == "admission":
        return f"admitted via {c.get('admission_type', '')}"
    return str(c)[:80]


def render_timeline(doc: dict, hide_discharge: bool = True,
                    max_day: int | None = None, *, audit: dict | None = None) -> str:
    """Day-grouped compact timeline. B1 hides the discharge summary and coded
    diagnoses (reference leakage). B3 additionally enforces result availability
    before the end of Day X and removes future stop/end/out dates from a copy.
    Optional audit receives aggregate record/field counts, not clinical text.
    """
    cutoff = prediction_cutoff(doc, max_day) if max_day is not None else None
    counts = defaultdict(int)
    by_day = defaultdict(list)
    for r in doc["records"]:
        counts["records_total"] += 1
        if (hide_discharge or cutoff is not None) and r["kind"] in ("note_ds", "diagnosis"):
            counts["hidden_reference"] += 1
            continue
        if max_day is not None and (r["day"] is None or r["day"] > max_day):
            counts["outside_event_window"] += 1
            continue
        if cutoff is not None:
            if r["kind"] in RESULT_KINDS:
                visible, reason = result_visible(r, cutoff)
                if not visible:
                    counts[f"excluded_{r['kind']}_{reason}"] += 1
                    continue
            content = dict(r["content"])
            for field in ("stop_day", "end_day", "out_day"):
                value = content.get(field)
                if value is not None and (isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not value <= max_day):
                    content.pop(field, None)
                    counts[f"masked_{field}"] += 1
            r = {**r, "content": content}
        counts["records_eligible"] += 1
        by_day[r["day"] if r["day"] is not None else "unknown"].append(r)

    m = doc["meta"]
    lines = [f"Patient: {m.get('gender') or '?'}/{m.get('age') or '?'}y, "
             f"admission type {m.get('admission_type')}, Day 0 = admission"]
    for day in sorted(by_day, key=lambda x: (not isinstance(x, int), x)):
        lines.append(f"=== Day {day} ===")
        rows = by_day[day]
        # labs: abnormal first, normals summarized
        labs = [r for r in rows if r["kind"] == "lab"]
        abnormal = [r for r in labs
                    if (r["content"].get("flag") or "").lower() in _ABNORMAL]
        normals = [r for r in labs if r not in abnormal]
        for r in abnormal[:config.RENDER_MAX_ABNORMAL_LABS]:
            lines.append(f"[Lab] {_fmt(r)}")
        if normals:
            names = sorted({r["content"]["name"] for r in normals})
            lines.append(f"[Lab] {len(normals)} normal results "
                         f"({', '.join(names[:8])}...)" if len(names) > 8 else
                         f"[Lab] {len(normals)} normal results ({', '.join(names)})")
        # emar: collapse to distinct meds of the day
        emar = [r for r in rows if r["kind"] == "emar"]
        if emar:
            meds = sorted({(r["content"].get("medication") or "?")
                           for r in emar if r["content"].get("medication")})
            lines.append(f"[MedAdmin] administered: {', '.join(meds[:15])}"
                         + (" ..." if len(meds) > 15 else ""))
        for r in rows:
            if r["kind"] in ("lab", "emar"):
                continue
            lines.append(f"[{_KIND_HEAD.get(r['kind'], r['kind'])}] {_fmt(r)}")
    if audit is not None:
        audit.update(counts)
    return "\n".join(lines)
