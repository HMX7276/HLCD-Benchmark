# -*- coding: utf-8 -*-
"""MIMIC → 统一时间线（延续既有 schema：records[{day, anchored_time,
granularity, source, kind, content}]，Day 0 = admittime）。"""

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.mimic.ds_sections import parse_discharge, parse_radiology
from src.tasks.temporal import TIMELINE_SCHEMA_VERSION, result_availability

IN = config.MIMIC_RAW
OUT = config.TIMELINE_DIR


def _day(ts, admit):
    if pd.isna(ts) or ts is None:
        return None
    return int((pd.Timestamp(ts) - admit).total_seconds() // 86400)


def _rec(day, at, gran, source, kind, content, *, availability=None):
    record = {"day": day, "anchored_time": str(at) if at is not None and not pd.isna(at)
            else "", "granularity": gran, "source": source, "kind": kind,
            "content": content}
    if availability is not None:
        record["availability"] = availability
    return record


def load_tables() -> dict:
    t = {}
    for name in ["cohort", "patients", "diagnoses_icd", "procedures_icd",
                 "prescriptions", "transfers", "services", "labevents",
                 "microbiologyevents", "emar", "procedureevents_icu",
                 "vitalsign", "gcs", "discharge", "radiology", "icustays",
                 "inputevents_blood"]:
        p = f"{IN}/{name}.csv"
        t[name] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
        for col in t[name].columns:
            if col.endswith("time") or col.endswith("date"):
                t[name][col] = pd.to_datetime(t[name][col], errors="coerce")
    return t


def build_one(hadm: int, t: dict) -> dict | None:
    coh = t["cohort"]
    row = coh[coh.hadm_id == hadm]
    if row.empty:
        return None
    row = row.iloc[0]
    admit = row.admittime
    subject = int(row.subject_id)
    pat = t["patients"][t["patients"].subject_id == subject]
    pat = pat.iloc[0] if not pat.empty else {}
    age = None
    if "anchor_age" in pat and not pd.isna(pat.get("anchor_age")):
        age = int(pat["anchor_age"] + (admit.year - pat["anchor_year"]))

    meta = {"hadm_id": hadm, "subject_id": subject,
            "admit_time": str(admit), "discharge_time": str(row.dischtime),
            "los_days": float((pd.Timestamp(row.dischtime) - admit).total_seconds() / 86400),
            "death_time": (None if pd.isna(row.get("deathtime")) else str(row.get("deathtime"))),
            "death": int(row.hospital_expire_flag),
            "admission_type": row.admission_type, "has_icu": int(row.has_icu),
            "gender": None if isinstance(pat.get("gender"), float) else pat.get("gender"),
            "age": age}

    recs = [_rec(0, admit, "exact", "cohort", "admission",
                 {"admission_type": row.admission_type,
                  "insurance": row.insurance, "ethnicity": row.ethnicity})]

    lab = t["labevents"][t["labevents"].hadm_id == hadm]
    for r in lab.itertuples():
        flag = None if pd.isna(r.flag) else r.flag
        recs.append(_rec(
            _day(r.charttime, admit), r.charttime, "exact",
            {"table": "labevents", "itemid": r.itemid}, "lab",
            {"name": r.label, "value": None if pd.isna(r.value) else str(r.value),
             "unit": None if pd.isna(r.valueuom) else r.valueuom,
             "flag": flag, "priority": None if pd.isna(r.priority) else r.priority},
            availability=result_availability(r.charttime, getattr(r, "storetime", None))))

    rx = t["prescriptions"]
    for r in (rx[rx.hadm_id == hadm].itertuples() if not rx.empty else []):
        recs.append(_rec(
            _day(r.starttime, admit), r.starttime, "exact",
            {"table": "prescriptions", "pharmacy_id": r.pharmacy_id}, "med",
            {"drug": r.drug,
             "dose": (f"{r.dose_val_rx} {r.dose_unit_rx}"
                      if not pd.isna(r.dose_val_rx) else None),
             "route": None if pd.isna(r.route) else r.route,
             "stop_day": _day(r.stoptime, admit)}))

    em = t["emar"]
    if not em.empty:
        for r in em[em.hadm_id == hadm].itertuples():
            recs.append(_rec(
                _day(r.charttime, admit), r.charttime, "exact",
                {"table": "emar", "emar_id": r.emar_id}, "emar",
                {"medication": None if pd.isna(r.medication) else r.medication,
                 "event": None if pd.isna(r.event_txt) else r.event_txt}))

    pr = t["procedures_icd"]
    for r in (pr[pr.hadm_id == hadm].itertuples() if not pr.empty else []):
        ts = r.chartdate
        recs.append(_rec(
            _day(ts, admit), ts, "day", {"table": "procedures_icd"}, "procedure",
            {"name": r.long_title if not pd.isna(r.long_title) else r.icd_code,
             "icd_code": r.icd_code, "icd_version": int(r.icd_version)}))

    pe = t["procedureevents_icu"]
    if not pe.empty:
        pe = pe.merge(t["icustays"][["stay_id", "hadm_id"]], on="stay_id")
        for r in pe[pe.hadm_id == hadm].itertuples():
            recs.append(_rec(
                _day(r.starttime, admit), r.starttime, "exact",
                {"table": "procedureevents_icu", "itemid": r.itemid},
                "icu_procedure",
                {"name": r.label, "value": None if pd.isna(r.value) else str(r.value),
                 "end_day": _day(r.endtime, admit)}))

    tr = t["transfers"]
    for r in (tr[tr.hadm_id == hadm].itertuples() if not tr.empty else []):
        recs.append(_rec(
            _day(r.intime, admit), r.intime, "exact", {"table": "transfers"},
            "transfer", {"eventtype": r.eventtype,
                         "careunit": None if pd.isna(r.careunit) else r.careunit,
                         "out_day": _day(r.outtime, admit)}))

    mb = t["microbiologyevents"]
    for r in (mb[mb.hadm_id == hadm].itertuples() if not mb.empty else []):
        ts = r.charttime if not pd.isna(r.charttime) else r.chartdate
        recs.append(_rec(
            _day(ts, admit), ts, "exact" if not pd.isna(r.charttime) else "day",
            {"table": "microbiologyevents"}, "microbio",
            {"test": r.test_name,
             "org": None if pd.isna(r.org_name) else r.org_name,
             "interpretation": (None if pd.isna(r.interpretation)
                                else r.interpretation)},
            availability=result_availability(ts, getattr(r, "storetime", None),
                                             getattr(r, "storedate", None))))

    rad = t["radiology"]
    if not rad.empty:
        for r in rad[rad.hadm_id == hadm].itertuples():
            sec = parse_radiology(r.text)
            recs.append(_rec(
                _day(r.charttime, admit), r.charttime, "exact",
                {"table": "radiology", "note_id": r.note_id}, "radiology",
                {"exam": sec.get("examination", "")[:120],
                 "findings": sec.get("findings", "")[:1200],
                 "impression": sec.get("impression", "")[:400]},
                availability=result_availability(r.charttime, getattr(r, "storetime", None))))

    # 每日生命体征 / GCS 中位数 / 血制品输注
    icu_stays = t["icustays"]
    if not icu_stays.empty:
        stays = icu_stays[icu_stays.hadm_id == hadm].stay_id.tolist()
        bl = t.get("inputevents_blood", pd.DataFrame())
        if not bl.empty and stays:
            for r in bl[bl.stay_id.isin(stays)].itertuples():
                recs.append(_rec(
                    _day(r.starttime, admit), r.starttime, "exact",
                    {"table": "inputevents"}, "transfusion",
                    {"product": r.itemid, "amount": (None if pd.isna(r.amount)
                                                     else str(r.amount)),
                     "unit": None if pd.isna(r.amountuom) else r.amountuom}))
        vs = t["vitalsign"]
        if not vs.empty and stays:
            v = vs[vs.stay_id.isin(stays)].copy()
            v["day"] = v.charttime.map(lambda x: _day(x, admit))
            v = v.dropna(subset=["day"])
            for day, g in v.groupby("day"):
                vals = {}
                for c in ["heart_rate", "sbp", "dbp", "resp_rate", "temperature",
                          "o2sat"]:
                    if c in g and g[c].notna().any():
                        vals[c] = float(g[c].median())
                if vals:
                    recs.append(_rec(int(day), None, "day",
                                     {"table": "vitalsign"}, "vitals", vals))
        gc = t["gcs"]
        if not gc.empty and stays:
            g2 = gc[gc.stay_id.isin(stays)].copy()
            g2["day"] = g2.charttime.map(lambda x: _day(x, admit))
            g2 = g2.dropna(subset=["day"])
            for day, g in g2.groupby("day"):
                if g.gcs.notna().any():
                    recs.append(_rec(int(day), None, "day", {"table": "gcs"},
                                     "gcs", {"min_gcs": float(g.gcs.min())}))

    # 出院小结（隐藏，B1 参考源）
    dis = t["discharge"]
    if not dis.empty:
        for r in dis[dis.hadm_id == hadm].itertuples():
            recs.append(_rec(
                _day(r.charttime, admit), r.charttime, "exact",
                {"table": "discharge", "note_id": r.note_id}, "note_ds",
                {"sections": parse_discharge(r.text), "text": r.text}))
        # 出院诊断 ICD（结构化对照）
    dx = t["diagnoses_icd"]
    if not dx.empty:
        for r in dx[dx.hadm_id == hadm].sort_values("seq_num").itertuples():
            recs.append(_rec(
                None, None, "phase", {"table": "diagnoses_icd"}, "diagnosis",
                {"name": r.long_title if not pd.isna(r.long_title) else r.icd_code,
                 "icd_code": r.icd_code, "seq": int(r.seq_num)}))

    recs.sort(key=lambda x: (x["day"] is None, x["day"] if x["day"] is not None else 0,
                             x["kind"]))
    return {"timeline_schema_version": TIMELINE_SCHEMA_VERSION,
            "admission_id": f"A-{hadm}", "patient_id": f"P-{subject}",
            "meta": meta, "records": recs}


def main():
    os.makedirs(OUT, exist_ok=True)
    t = load_tables()
    n, anchored, total = 0, 0, 0
    for hadm in sorted(t["cohort"].hadm_id.tolist()):
        doc = build_one(int(hadm), t)
        if doc is None:
            continue
        with open(f"{OUT}/{doc['admission_id']}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        n += 1
        anchored += sum(1 for r in doc["records"]
                        if r["granularity"] in ("exact", "day"))
        total += len(doc["records"])
    print(f"timelines: {n} -> {OUT}")
    print(f"records: {total:,}, anchored {anchored/total:.1%}")


if __name__ == "__main__":
    main()
