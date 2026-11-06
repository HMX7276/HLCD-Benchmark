# -*- coding: utf-8 -*-
"""run_concurrent: ThreadPool driver shared by the LLM runners."""
import time
import unittest
from unittest import mock

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.llm.parallel import run_concurrent


class TestRunConcurrent(unittest.TestCase):
    def test_results_in_input_order(self):
        out = run_concurrent(lambda x: x * 2, list(range(20)))
        self.assertEqual([r for r, _ in out], [x * 2 for x in range(20)])
        self.assertTrue(all(e is None for _, e in out))

    def test_error_isolated_not_fatal(self):
        def w(x):
            if x % 3 == 0:
                raise ValueError("boom")
            return x

        out = run_concurrent(w, list(range(10)))
        self.assertIsNone(out[3][0])
        self.assertIsInstance(out[3][1], ValueError)
        self.assertEqual(out[1], (1, None))
        self.assertEqual(out[4], (4, None))

    def test_parallel_speedup(self):
        def w(x):
            time.sleep(0.3)
            return x

        with mock.patch.object(config, "LLM_CONCURRENCY", 4):
            t0 = time.monotonic()
            out = run_concurrent(w, list(range(8)))
            dt = time.monotonic() - t0
        self.assertEqual([r for r, _ in out], list(range(8)))
        # sequential would take 8 * 0.3 = 2.4 s; allow generous CI margin
        self.assertLess(dt, 1.8)

    def test_progress_callback_counts_and_indexes(self):
        seen = []
        run_concurrent(lambda x: x, ["a", "b", "c"],
                       lambda i, r, e, done, total: seen.append((i, r, e, done, total)))
        self.assertEqual([s[3] for s in seen], [1, 2, 3])
        self.assertTrue(all(s[4] == 3 for s in seen))
        by_idx = {s[0]: s[1] for s in seen}
        self.assertEqual(by_idx, {0: "a", 1: "b", 2: "c"})

    def test_empty_items(self):
        self.assertEqual(run_concurrent(lambda x: x, []), [])


if __name__ == "__main__":
    unittest.main()
