# -*- coding: utf-8 -*-
"""MIMIC cohort extraction (read-only, no DB writes):
1) export the discharge-note hadm list + small tables from PostgreSQL;
2) stratified cohort sampling locally (exact proportional quotas);
3) bulk COPY-export all event tables for the cohort -> $HLCD_WORK/mimic/*.csv

Everything (cohort size, seed, min LOS, connection) is configured via `.env`.
"""
import io
import os
import random

import pandas as pd
import psycopg2

from src import config


def conn(db: str):
    return psycopg2.connect(dbname={"mimiciv": config.PG_DB_MIMICIV,
                                     "mimicnote": config.PG_DB_MIMICNOTE}.get(db, db),
                            **config.PG)


def copy_df(cnx, sql: str) -> pd.DataFrame:
    buf = io.StringIO()
    with cnx.cursor() as cur:
        cur.copy_expert(f"COPY ({sql}) TO STDOUT WITH CSV HEADER", buf)
    return pd.read_csv(io.StringIO(buf.getvalue()))


def id_array(ids) -> str:
    """Postgres array literal for `= ANY (...)` filters (safe for large cohorts)."""
    return "(ARRAY[" + ",".join(str(int(i)) for i in ids) + "]::bigint[])"


def exact_quota(strata_sizes: dict, total: int) -> dict:
    """Largest-remainder proportional allocation; caps at stratum size."""
    n = sum(strata_sizes.values())
    if total >= n:
        return dict(strata_sizes)
    raw = {s: size * total / n for s, size in strata_sizes.items()}
    quota = {s: int(v) for s, v in raw.items()}
    for s, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]),
                       reverse=True)[:total - sum(quota.values())]:
        quota[s] += 1
    return {s: min(q, strata_sizes[s]) for s, q in quota.items()}


def main():
    out = config.MIMIC_RAW
    os.makedirs(out, exist_ok=True)

    # ---- 1. small tables + note hadm list ----
    cnv = conn("mimiciv")
    cnn = conn("mimicnote")
    ds_hadms = copy_df(cnn, "SELECT DISTINCT hadm_id FROM mimiciv_note.discharge")
    adm = copy_df(cnv, """SELECT hadm_id, subject_id, admittime, dischtime,
        deathtime, admission_type, hospital_expire_flag, insurance,
        marital_status, race AS ethnicity FROM mimiciv_hosp.admissions""")
    icu = copy_df(cnv, """SELECT stay_id, subject_id, hadm_id, intime, outtime,
        first_careunit, last_careunit FROM mimiciv_icu.icustays""")
    cnn.close()

    # ---- 2. stratified sampling (exact quotas, no ordering bias) ----
    adm = adm[adm.hadm_id.isin(ds_hadms.hadm_id)].copy()
    adm["los_days"] = (pd.to_datetime(adm.dischtime)
                       - pd.to_datetime(adm.admittime)).dt.total_seconds() / 86400
    adm = adm[adm.los_days >= config.MIN_LOS_DAYS]
    adm = adm.merge(icu[["hadm_id", "stay_id"]].drop_duplicates("hadm_id"),
                    on="hadm_id", how="left")
    adm["has_icu"] = adm.stay_id.notna().astype(int)
    adm["stratum"] = (adm.admission_type + "|icu" + adm.has_icu.astype(str)
                      + "|exp" + adm.hospital_expire_flag.astype(str))
    rng = random.Random(config.COHORT_SEED)
    sizes = adm.groupby("stratum").size().to_dict()
    if config.COHORT_SIZE <= 0 or len(adm) < config.COHORT_SIZE:
        cnv.close()
        raise ValueError(f"Requested {config.COHORT_SIZE} admissions but only {len(adm)} "
                         "are eligible; choose a positive size no greater than the pool")
    quota = exact_quota(sizes, config.COHORT_SIZE)
    cohort = []
    for s, take in quota.items():
        pool = adm[adm.stratum == s].hadm_id.tolist()
        cohort += rng.sample(pool, take)
    while len(cohort) < config.COHORT_SIZE:  # 某层不足时从剩余里补齐
        rest = adm[~adm.hadm_id.isin(cohort)].hadm_id.tolist()
        cohort += rng.sample(rest, config.COHORT_SIZE - len(cohort))
    cohort = sorted(cohort[:config.COHORT_SIZE])
    coh = adm[adm.hadm_id.isin(cohort)]
    coh[["hadm_id", "subject_id", "admittime", "dischtime", "deathtime",
         "admission_type", "hospital_expire_flag", "los_days", "has_icu",
         "insurance", "marital_status", "ethnicity"]].to_csv(
        f"{out}/cohort.csv", index=False)
    print(f"cohort: {len(coh)}/{config.COHORT_SIZE} (strata {coh.stratum.nunique()}, "
          f"ICU {coh.has_icu.mean():.0%}, expired {coh.hospital_expire_flag.mean():.0%}, "
          f"LOS median {coh.los_days.median():.1f}d)")

    H = id_array(cohort)
    S = id_array(icu[icu.hadm_id.isin(coh.hadm_id)].stay_id.dropna())
    icu[icu.hadm_id.isin(cohort)].to_csv(f"{out}/icustays.csv", index=False)

    # ---- 3. bulk export ----
    jobs = [
        ("patients", f"""SELECT * FROM mimiciv_hosp.patients WHERE subject_id IN
            (SELECT subject_id FROM mimiciv_hosp.admissions WHERE hadm_id = ANY {H})"""),
        ("diagnoses_icd", f"""SELECT d.hadm_id, d.seq_num, d.icd_code,
            d.icd_version, t.long_title FROM mimiciv_hosp.diagnoses_icd d
            LEFT JOIN mimiciv_hosp.d_icd_diagnoses t
            ON d.icd_code = t.icd_code AND d.icd_version = t.icd_version
            WHERE d.hadm_id = ANY {H}"""),
        ("procedures_icd", f"""SELECT p.hadm_id, p.seq_num, p.chartdate,
            p.icd_code, p.icd_version, t.long_title
            FROM mimiciv_hosp.procedures_icd p
            LEFT JOIN mimiciv_hosp.d_icd_procedures t
            ON p.icd_code = t.icd_code AND p.icd_version = t.icd_version
            WHERE p.hadm_id = ANY {H}"""),
        ("prescriptions", f"""SELECT hadm_id, pharmacy_id, drug, dose_val_rx,
            dose_unit_rx, route, starttime, stoptime
            FROM mimiciv_hosp.prescriptions WHERE hadm_id = ANY {H}"""),
        ("transfers", f"""SELECT hadm_id, eventtype, careunit, intime, outtime
            FROM mimiciv_hosp.transfers WHERE hadm_id = ANY {H}"""),
        ("services", f"""SELECT hadm_id, prev_service, curr_service, transfertime
            FROM mimiciv_hosp.services WHERE hadm_id = ANY {H}"""),
        ("labevents", f"""SELECT e.hadm_id, e.charttime, e.storetime, e.itemid,
            i.label, i.fluid, e.value, e.valueuom, e.flag, e.priority
            FROM mimiciv_hosp.labevents e
            JOIN mimiciv_hosp.d_labitems i ON e.itemid = i.itemid
            WHERE e.hadm_id = ANY {H}""", "labevents full-table scan"),
        ("microbiologyevents", f"""SELECT hadm_id, chartdate, charttime, storedate, storetime,
            spec_type_desc, test_name, org_name, ab_name, interpretation, quantity
            FROM mimiciv_hosp.microbiologyevents WHERE hadm_id = ANY {H}"""),
        ("emar", f"""SELECT hadm_id, emar_id, charttime, medication, event_txt
            FROM mimiciv_hosp.emar WHERE hadm_id = ANY {H}"""),
    ]
    for job in jobs:
        name, sql = job[0], job[1]
        note = job[2] if len(job) > 2 else ""
        print(f"export {name}..." + (f" [{note}]" if note else ""), flush=True)
        df = copy_df(cnv, sql)
        df.to_csv(f"{out}/{name}.csv", index=False)
        print(f"  -> {len(df):,} rows")

    icu_jobs = [
        ("procedureevents_icu", f"""SELECT e.stay_id, e.starttime, e.endtime,
            e.itemid, i.label, e.value, e.valueuom
            FROM mimiciv_icu.procedureevents e
            JOIN mimiciv_icu.d_items i ON e.itemid = i.itemid
            WHERE e.stay_id = ANY {S}"""),
        ("inputevents_blood", f"""SELECT e.stay_id, e.starttime, e.endtime,
            e.itemid, i.label, e.amount, e.amountuom
            FROM mimiciv_icu.inputevents e
            JOIN mimiciv_icu.d_items i ON e.itemid = i.itemid
            WHERE e.stay_id = ANY {S}
              AND (i.category = 'Blood Products' OR i.label ILIKE ANY (ARRAY[
                   '%packed red%', '%platelet%', '%fresh frozen%',
                   '%cryoprecipitate%']))"""),
        ("vitalsign", f"""SELECT v.stay_id, v.charttime, v.heart_rate, v.sbp,
            v.dbp, v.mbp, v.resp_rate, v.temperature, v.spo2 AS o2sat
            FROM mimiciv_derived.vitalsign v WHERE v.stay_id = ANY {S}"""),
        ("gcs", f"""SELECT stay_id, charttime, gcs, gcs_motor, gcs_verbal,
            gcs_eyes FROM mimiciv_derived.gcs WHERE stay_id = ANY {S}"""),
    ]
    for name, sql in icu_jobs:
        print(f"export {name}...", flush=True)
        df = copy_df(cnv, sql)
        df.to_csv(f"{out}/{name}.csv", index=False)
        print(f"  -> {len(df):,} rows")
    cnv.close()

    cnn = conn("mimicnote")
    for name, sql in [
        ("discharge", f"""SELECT note_id, subject_id, hadm_id, note_seq,
            charttime, storetime, text FROM mimiciv_note.discharge
            WHERE hadm_id = ANY {H}"""),
        ("radiology", f"""SELECT note_id, subject_id, hadm_id, note_seq,
            charttime, storetime, text FROM mimiciv_note.radiology
            WHERE hadm_id = ANY {H}"""),
    ]:
        print(f"export {name}...", flush=True)
        df = copy_df(cnn, sql)
        df.to_csv(f"{out}/{name}.csv", index=False)
        print(f"  -> {len(df):,} rows")
    cnn.close()
    print("DONE ->", out)


if __name__ == "__main__":
    main()
