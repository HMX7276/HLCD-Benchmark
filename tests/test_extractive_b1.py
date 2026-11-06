# -*- coding: utf-8 -*-
"""Pure-function tests for the extractive B1 baseline (no LLM / DB)."""
import unittest

from src.baselines.extractive_b1 import build_prediction, parse_timeline
from src.eval.metrics import b1_entity_metrics

TIMELINE = """Patient: M/64y, admission type SURGICAL, Day 0 = admission
=== Day -1 ===
[Procedure] Drainage of peritoneal cavity (open)
[Lab] Sodium = 127 [abnormal] mEq/L
[Medication] Heparin 5000 units IV (stop Day 3)
=== Day 0 ===
[Admission] admitted via SURGICAL
[Lab] Potassium = 3.0 [l] mmol/L
[Lab] Glucose = 100 mg/dL
[Medication] Vancomycin (IV) 15 mg/kg
[MedAdmin] administered: Acetaminophen, Ondansetron
[Transfer] admit -> MICU
=== Day 1 ===
[Vitals] HR=90.0 SBP=120.0 DBP=70.0
[GCS] min GCS 15
[ICUProcedure] Extubation
[Radiology] Chest X-ray: bilateral effusions, no pneumothorax
=== Day 4 ===
[Transfer] ward discharge/disposition event
"""


class TestParseTimeline(unittest.TestCase):
    def test_days_and_kinds(self):
        rows = parse_timeline(TIMELINE)
        days = {r["day"] for r in rows}
        self.assertEqual(days, {-1, 0, 1, 4})
        kinds = {r["kind"] for r in rows}
        self.assertIn("Lab", kinds)
        self.assertIn("Medication", kinds)
        self.assertIn("Radiology", kinds)
        self.assertIn("Transfer", kinds)
        self.assertNotIn("Patient", kinds)  # header line is not a [Kind] row

    def test_document_order(self):
        rows = parse_timeline(TIMELINE)
        labs = [r for r in rows if r["kind"] == "Lab"]
        self.assertEqual([r["day"] for r in labs], [-1, 0, 0])


class TestBuildPrediction(unittest.TestCase):
    def setUp(self):
        self.pred = build_prediction(TIMELINE)

    def test_medications(self):
        meds = {m["name"]: m["day"] for m in self.pred["medications"]}
        self.assertEqual(meds.get("Heparin"), -1)
        self.assertEqual(meds.get("Acetaminophen"), 0)
        self.assertEqual(meds.get("Ondansetron"), 0)
        self.assertEqual(meds.get("Vancomycin"), 0)  # parens+dose stripped
        self.assertNotIn("Glucose", meds)

    def test_surgeries(self):
        surgs = {s["name"]: s["day"] for s in self.pred["surgeries"]}
        self.assertEqual(surgs.get("Drainage of peritoneal cavity (open)"), -1)
        self.assertEqual(surgs.get("Extubation"), 1)

    def test_key_abnormal_labs_only(self):
        labs = {l["name"]: l for l in self.pred["key_abnormal_labs"]}
        self.assertEqual(set(labs), {"Sodium", "Potassium"})
        self.assertEqual(labs["Sodium"]["value"], "127 mEq/L (abnormal)")
        self.assertEqual(labs["Potassium"]["value"], "3.0 mmol/L (low)")
        self.assertEqual(labs["Potassium"]["day"], 0)

    def test_imaging(self):
        imgs = self.pred["key_imaging_findings"]
        self.assertEqual(len(imgs), 1)
        self.assertEqual(imgs[0]["name"],
                         "bilateral effusions, no pneumothorax")
        self.assertEqual(imgs[0]["day"], 1)

    def test_structurally_unreachable_entities(self):
        self.assertEqual(self.pred["diagnoses"], [])
        self.assertEqual(self.pred["allergy"], "")

    def test_outcome(self):
        self.assertEqual(self.pred["outcome"],
                         "The patient was discharged on Day 4.")

    def test_scoreable(self):
        ref = [{"type": "discharge_medication", "name": "Heparin"},
               {"type": "discharge_medication", "name": "Ondansetron"},
               {"type": "procedure", "name": "Extubation", "day": 1},
               {"type": "discharge_diagnosis", "name": "Adynamic ileus"}]
        m = b1_entity_metrics(ref, self.pred)
        self.assertAlmostEqual(m["entity_recall"], 0.75)
        self.assertAlmostEqual(m["per_type"]["discharge_medication"]["recall"],
                               1.0)
        self.assertAlmostEqual(m["per_type"]["discharge_diagnosis"]["recall"],
                               0.0)  # diagnoses unreachable by design


if __name__ == "__main__":
    unittest.main()
