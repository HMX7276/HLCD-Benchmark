# -*- coding: utf-8 -*-
"""ThreadPool driver for the LLM runners: LLM_CONCURRENCY from .env (1 = serial)."""
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import config


def run_concurrent(worker, items, on_done=None):
    """Run worker(item) over items with LLM_CONCURRENCY threads.

    A worker exception is captured as (None, err) and does not stop the run.
    Returns [(result, err), ...] in input order so outputs are bit-identical
    to the sequential version. on_done(i, result, err, done, total) fires in
    the main thread on each completion (progress printing).
    """
    total = len(items)
    out = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as ex:
        futs = {ex.submit(worker, it): i for i, it in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                r, e = fut.result(), None
            except Exception as err:
                r, e = None, err
            out[i] = (r, e)
            done += 1
            if on_done:
                on_done(i, r, e, done, total)
    return out
