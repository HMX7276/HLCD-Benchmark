"""Synthetic, offline regression tests for B3 temporal availability."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.mimic.timeline import build_one
from src.tasks.b3 import derive_events
from src.tasks.render import render_timeline
from src.tasks.temporal import (B3_INPUT_POLICY, prediction_cutoff,
                                result_availability, result_visible)
from src.tasks.b3_provenance import validate_timeline
from src.runners import run_b3
from src import config
from src.tasks.b3 import EVENTS


ADMIT = datetime(2100, 1, 1, 10)


def synthetic_tables(stored=None, stop_day=4):
    event = ADMIT + timedelta(days=1, hours=1)
    stored = stored if stored is not None else event
    names = ('cohort patients labevents prescriptions emar procedures_icd '
             'procedureevents_icu transfers microbiologyevents radiology '
             'icustays discharge diagnoses_icd vitalsign gcs').split()
    tables = {n: pd.DataFrame() for n in names}
    tables['cohort'] = pd.DataFrame([dict(hadm_id=1, subject_id=1, admittime=ADMIT,
        dischtime=ADMIT+timedelta(days=6), los_days=6, hospital_expire_flag=0,
        admission_type='SYNTHETIC', has_icu=0, insurance='SYNTHETIC', ethnicity='SYNTHETIC')])
    tables['patients'] = pd.DataFrame([dict(subject_id=1, gender='F', anchor_age=50, anchor_year=2100)])
    tables['labevents'] = pd.DataFrame([dict(hadm_id=1, charttime=event, storetime=stored,
        itemid=999, label='TEST_LAB', value='123', valueuom='unit', flag='abnormal', priority='ROUTINE')])
    tables['prescriptions'] = pd.DataFrame([dict(hadm_id=1, starttime=event,
        stoptime=ADMIT+timedelta(days=stop_day), pharmacy_id=1,
        drug='vancomycin', dose_val_rx='1', dose_unit_rx='g', route='IV')])
    tables['microbiologyevents'] = pd.DataFrame([dict(hadm_id=1, charttime=event,
        chartdate=event.replace(hour=0), storetime=stored, storedate=pd.NaT,
        test_name='TEST_CULTURE', org_name='TEST_ORGANISM', interpretation='S')])
    tables['radiology'] = pd.DataFrame([dict(hadm_id=1, charttime=event, storetime=stored,
        note_id='SYNTHETIC', text='EXAMINATION: TEST_SCAN\nIMPRESSION: TEST_FINDING')])
    return tables


class TestPipeline(unittest.TestCase):
    def assert_results(self, text, present):
        for marker in ('TEST_LAB', 'TEST_ORGANISM', 'TEST_FINDING'):
            with self.subTest(marker=marker):
                self.assertEqual(marker in text, present)

    def test_late_results_and_future_stops(self):
        doc = build_one(1, synthetic_tables(ADMIT + timedelta(days=4)))
        original = deepcopy(doc)
        for x in (1, 2, 3):
            with self.subTest(X=x):
                audit = {}
                text = render_timeline(doc, max_day=x, audit=audit)
                self.assertNotIn('stop Day 4', text)
                self.assert_results(text, False)
                self.assertEqual(audit['excluded_lab_available_after_cutoff'], 1)
        self.assertEqual(doc, original)
        self.assertIn(4, derive_events(doc)['antibiotic_stop'])
        # B1 behavior and full label source remain intact.
        self.assertIn('stop Day 4', render_timeline(doc))
        self.assert_results(render_timeline(doc), True)
        self.assert_results(render_timeline(doc, max_day=4), True)

    def test_direct_target_label_hidden_but_preserved(self):
        for x in (1, 2, 3):
            doc = build_one(1, synthetic_tables(stop_day=x+1))
            self.assertNotIn(f'stop Day {x+1}', render_timeline(doc, max_day=x))
            self.assertIn(x+1, derive_events(doc)['antibiotic_stop'])

    def test_exact_cutoff_is_exclusive_nonmidnight_admission(self):
        cutoff = ADMIT + timedelta(days=2)
        for delta, visible in ((-1, True), (0, False), (1, False)):
            with self.subTest(seconds=delta):
                doc = build_one(1, synthetic_tables(cutoff+timedelta(seconds=delta)))
                self.assert_results(render_timeline(doc, max_day=1), visible)

    def test_missing_result_times_fail_closed(self):
        doc = build_one(1, synthetic_tables(pd.NaT))
        self.assert_results(render_timeline(doc, max_day=1), False)
        # Legacy row metadata cannot accidentally bypass filtering.
        for r in doc['records']:
            r.pop('availability', None)
        self.assert_results(render_timeline(doc, max_day=1), False)

    def test_legacy_csv_columns_missing_still_fail_closed(self):
        tables = synthetic_tables()
        for table in ('labevents', 'microbiologyevents', 'radiology'):
            tables[table] = tables[table].drop(columns='storetime')
        self.assert_results(render_timeline(build_one(1, tables), max_day=1), False)

    def test_date_only_microbiology_conservative(self):
        tables = synthetic_tables(pd.NaT)
        tables['microbiologyevents']['storedate'] = datetime(2100, 1, 3)
        doc = build_one(1, tables)
        self.assertNotIn('TEST_ORGANISM', render_timeline(doc, max_day=1))  # Jan 3 10:00 cutoff
        self.assertIn('TEST_ORGANISM', render_timeline(doc, max_day=2))
        r = next(r for r in doc['records'] if r['kind'] == 'microbio')
        self.assertTrue(result_visible(r, datetime(2100, 1, 4))[0])

    def test_before_event_and_invalid_times_fail_closed(self):
        for stored in (ADMIT, 'bad timestamp', '2100-01-02T12:00:00+00:00'):
            doc = build_one(1, synthetic_tables(stored))
            self.assert_results(render_timeline(doc, max_day=1), False)

    def test_normal_lab_names_do_not_leak(self):
        tables = synthetic_tables(ADMIT+timedelta(days=4))
        tables['labevents']['flag'] = None
        self.assertNotIn('TEST_LAB', render_timeline(build_one(1, tables), max_day=1))

    def test_future_event_day_and_hidden_reference(self):
        doc = build_one(1, synthetic_tables())
        for r in doc['records']:
            if r['kind'] != 'admission':
                r['day'] = 4
        doc['records'].append(dict(day=1, kind='note_ds', content={'text': 'SECRET'}))
        text = render_timeline(doc, max_day=1, hide_discharge=False)
        self.assert_results(text, False)
        self.assertNotIn('vancomycin', text)
        self.assertNotIn('SECRET', text)

    def test_invalid_admission_or_day_is_error(self):
        doc = build_one(1, synthetic_tables())
        for x in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                prediction_cutoff(doc, x)
        doc['meta']['admit_time'] = ''
        with self.assertRaises(ValueError):
            render_timeline(doc, max_day=1)

    def test_schema_validation(self):
        doc = build_one(1, synthetic_tables())
        validate_timeline(doc)
        doc.pop('timeline_schema_version')
        with self.assertRaisesRegex(ValueError, 'rebuilt'):
            validate_timeline(doc)

    def test_malformed_exact_time_does_not_use_date_fallback(self):
        info = result_availability(ADMIT, 'broken', datetime(2100, 1, 2))
        self.assertNotEqual(info['status'], 'known')


if __name__=='__main__': unittest.main()
