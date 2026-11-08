# Local release validation

## v0.2.1 validation (2026-11-08)

- After removing automatic perturbation scoring: 94 tests passed, zero failures,
  errors or skips. Ten tests specific to the removed classifier were deleted;
  two runner checks were added for failed-condition preservation and empty input.
- Paired response persistence, execution counts and compatible checkpoint reuse
  are covered by mocked integration tests. No real model endpoint was called.
- Version 0.2.1 supersedes v0.2.0 for response collection; archived releases
  retain their original code and score definitions.

## Archived v0.2.0 validation

Checked on 2026-11-05 in a fresh Windows virtual environment installed from
this release's `requirements.txt`.

- Python 3.12.14; pandas 2.2.3; psycopg2-binary 2.9.12; python-dotenv 1.1.0.
- `python -m unittest discover -s tests -v`: 102 tests passed, zero errors,
  failures or skips. Model entry-point integration uses mocked model responses.
- `python scripts/check_public_release.py`: public-tree/credential-pattern checks passed.
- Git ignore checks: populated `.env` variants and data/output paths are ignored;
  `.env.example`, code, tests and the workflow are included.

No database connection, actual model call or patient-level dataset processing
was performed. These checks are software checks, not scientific validation of
historical benchmark results. GitHub-hosted CI has not yet run for this prepared
release. The provided CI workflow targets Python 3.11 and 3.12.

Version 0.2 removes consensus inference and changes response validation, labels,
scoring and checkpoint schemas. Regression tests cover the previously reproduced
failures, including actual runner entry points with simulated responses. Active
source and the public source tree are synchronized at this release. Historical
ZIPs remain separate and do not share this version's score/checkpoint semantics.
