# -*- coding: utf-8 -*-
"""English-edition unit tests: DS section split, metrics matching, traps."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import b1_entity_metrics, b3_multilabel_metrics, norm
from src.mimic.ds_sections import parse_discharge, parse_radiology
from src.traps.build import _flip_flag, _mutate_value, _shift_day

DS = """Name: ___
Sex: F
Allergies: No Known Allergies
Chief Complaint: Worsening abdominal distension
Major Surgical or Invasive Procedure: Paracentesis
HOSPITAL COURSE: The patient was admitted and treated with diuretics.
DISCHARGE DIAGNOSES: 1. Ascites 2. Cirrhosis
DISCHARGE MEDICATIONS:
Lasix 40 mg daily
Spironolactone 100 mg daily
"""

RAD = """EXAMINATION: CHEST PA
INDICATION: eval for effusion
FINDINGS: small bilateral effusions.
IMPRESSION: Pleural effusions.
"""


class TestSections(unittest.TestCase):
    def test_ds(self):
        s = parse_discharge(DS)
        self.assertIn("Worsening abdominal distension",
                      s["chief complaint"])
        self.assertIn("Paracentesis", s["major surgical or invasive procedure"])
        self.assertIn("diuretics", s["hospital course"])
        self.assertIn("1. Ascites", s["discharge diagnosis"])
        self.assertIn("Lasix", s["discharge medication"])

    def test_radiology(self):
        s = parse_radiology(RAD)
        self.assertIn("CHEST PA", s["examination"])
        self.assertIn("effusions", s["findings"])
        self.assertIn("Pleural", s["impression"])


class TestMetricsEn(unittest.TestCase):
    REF = [{"type": "discharge_diagnosis", "name": "Sepsis due to E. coli"},
           {"type": "procedure", "name": "paracentesis", "day": 2},
           {"type": "discharge_medication", "name": "Furosemide"},
           {"type": "allergy", "name": "Penicillin"}]

    def test_norm_keeps_spaces(self):
        self.assertEqual(norm("Acute Kidney  Injury."), "acute kidney injury")

    def test_word_boundary_match(self):
        pred = {"diagnoses": [{"name": "sepsis"}],  # 完整词，可边界匹配
                "surgeries": [{"name": "Paracentesis", "day": 2}],
                "medications": [{"name": "Furosemide 40 mg"}],
                "allergy": "penicillin"}
        m = b1_entity_metrics(self.REF, pred)
        self.assertEqual(m["per_type"]["discharge_diagnosis"]["recall"], 1.0)
        self.assertEqual(m["per_type"]["procedure"]["recall"], 1.0)
        self.assertEqual(m["per_type"]["discharge_medication"]["recall"], 1.0)
        self.assertEqual(m["time_alignment"], 1.0)

    def test_b3(self):
        yt = [{"a": 1, "b": 0}, {"a": 0, "b": 1}]
        yp = [{"a": 1, "b": 0}, {"a": 0, "b": 0}]  # a 完美, b 全漏
        m = b3_multilabel_metrics(yt, yp, ["a", "b"])
        self.assertAlmostEqual(m["macro_f1"], 0.5)


class TestTrapsEn(unittest.TestCase):
    TEXT = "=== Day 2 ===\n[Lab] Creatinine = 2.1 [abnormal] (mg/dL)\n" \
           "[Medication] Vancomycin 1 g IV\n=== Day 3 ===\n[Lab] Sodium = 139\n"

    def test_mutate(self):
        r = _mutate_value(self.TEXT, __import__("random").Random(1))
        self.assertIsNotNone(r)
        self.assertEqual(r[1]["type"], "value_magnitude")
        self.assertIn(r[1]["tampered"], r[0])

    def test_flip(self):
        r = _flip_flag(self.TEXT, __import__("random").Random(1))
        self.assertEqual(r[1]["type"], "polarity_flip")

    def test_shift(self):
        r = _shift_day(self.TEXT, __import__("random").Random(1))
        self.assertEqual(r[1]["type"], "temporal_shift")


if __name__ == "__main__":
    unittest.main()


class TestQuotaAndSplits(unittest.TestCase):
    def test_exact_quota(self):
        from src.mimic.extract import exact_quota
        q = exact_quota({"a": 600, "b": 300, "c": 100}, 100)
        self.assertEqual(sum(q.values()), 100)
        self.assertGreaterEqual(q["a"], 55)
        q2 = exact_quota({"a": 10}, 100)  # 池不足 -> 全取
        self.assertEqual(q2, {"a": 10})

    def test_split_ratio_parse(self):
        from src import config
        parts = [float(x) for x in "5:1:3".split(":")]
        self.assertAlmostEqual(sum(parts) / sum(parts), 1.0)
