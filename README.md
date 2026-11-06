# HLCD — Hospital Longitudinal Clinical Data

Research code for evaluating language models on longitudinal hospital records:
B1 summarization, B3 next-day event prediction, and experimental perturbation
scoring. This source release is prepared as **v0.2.0** under Apache-2.0.

**Status:** research software, not a clinical decision support system. This
release includes B3 temporal-input corrections, but does not contain rerun
model results or establish the validity of all benchmark metrics. Read
[known limitations](docs/KNOWN_LIMITATIONS.md) before interpreting scores.

## What is included

- Python extraction, timeline, task, scoring and model-runner code.
- Offline synthetic unit and integration tests; no model calls are required.
- Example configuration, temporal migration instructions and release checks.
- No patient data, admission lists, model transcripts, expert-review exports,
  model weights, manuscript files or historical experiment outputs.

## Install and test

Python 3.11 or 3.12 is recommended; the local release check used Python 3.12.
Run commands from the repository root. The package is used directly from its
checkout; it is not currently distributed as an installable PyPI package.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_public_release.py
```

On Linux/macOS, use `.venv/bin/python` in place of the Windows executable path.
The test suite uses synthetic inputs and mocked model outputs. It neither
requires MIMIC access nor validates real-dataset model performance.

## Configure an authorized experiment

Copy `.env.example` to `.env`, then supply the local database, model ID,
endpoint and credential settings. Examples are not historical run settings.
Do not commit the populated `.env` file. No model or database runs are launched
by installing dependencies or running the offline tests.

The pipeline expects MIMIC-IV `hosp`/`icu`, MIMIC-IV-Note discharge/radiology,
and day-level vitalsign/GCS inputs in the configured derived schema. The study
uses MIMIC-IV v3.1 and MIMIC-IV-Note v2.2. Derived tables are project inputs,
not raw tables supplied by those database releases; their construction must
be documented and supplied by the executing team. This repository does not
include the SQL that creates those derived tables.

The following commands process real data and/or invoke models. Run them only
in an appropriately authorized environment after configuration:

```text
python -m src.mimic.extract
python -m src.mimic.timeline
python -m src.tasks.b1
python -m src.tasks.splits
python -m src.tasks.b3
python -m src.runners.run_b1 --n 20 --out B1_RESULT.json
python -m src.runners.run_b3 --n 20 --X 1 --out B3_RESULT.json --ckpt B3_CHECKPOINT.jsonl
```

For an existing cohort, **do not run extraction as a migration shortcut**:
it samples a new cohort. Follow [fixed-cohort migration instructions](docs/B3_TEMPORAL_MIGRATION.txt).
The `--n` option is an upper limit after eligibility/split filtering, not proof
that the requested number of admissions was evaluated. Preserve the actual
case lists, provider model versions, run settings, failure counts and artifacts
in the authorized environment.

## Evaluation and score versions

Only single-model inference is included. Failures are counted separately and
excluded from valid-response metrics; they are never silently converted into
correct negative predictions. B1 lexical scoring (`b1-assertion-lexical-v2`)
checks assertion scope but remains a heuristic requiring expert validation.
B3 labels (`b3-exact-events-v2`) use exact discharge/death timestamps; cases
with unknown in-hospital death time are excluded and counted.

Perturbations use paired clean/tampered inputs with schema 2 target metadata.
`traps-target-aware-v2` reports explicit recognition, incorrect adoption,
not mentioned and unscorable states separately. Clean targeted-warning rates
are a control, not validated false-positive rates. `temporal_order_swap`
replaces the former text-order operation; no causal reasoning claim is made.
Legacy scores and checkpoints are not compatible with these changed rules.

Generate/evaluate perturbations in the authorized environment:

```text
python -m src.traps.build
python -m src.traps.evaluate --out TRAP_RESULT.json --ckpt TRAP_CHECKPOINT.jsonl
```

See [v0.2 migration notes](docs/MIGRATION_V0.2.md).

## B3 temporal policy

Timeline schema **3**, input policy **b3-availability-v1**:

- Day X ends at `admittime + (X+1)*24 hours`, with an exclusive exact-time cutoff.
- Future medication stop dates are removed only from the B3 input view; the
  full timeline remains available for event-label construction.
- Lab/radiology results require valid stored availability timestamps.
- Microbiology uses the last-result-update time; date-only availability is
  conservatively delayed to the next calendar midnight. Missing or invalid
  availability is excluded instead of falling back to observation time.
- Old timeline schemas and mismatched checkpoints are rejected. New outputs
  include input filtering counts and code/input/run fingerprints.

This corrects specified code paths, not every possible source of retrospective
information. Existing published or manuscript scores cannot be relabeled as
corrected results without the corresponding analysis and affected-input reruns.

## Data and logging

The code license grants no access to MIMIC or rights to redistribute it. Obtain
applicable access and follow the dataset terms through PhysioNet. Do not post
real records, derived patient text, individual model responses or reviewer
evidence in issues or pull requests. Use synthetic reproductions.

The model client writes response transcripts under `HLCD_WORK/mimic_runs`.
These logs can contain clinical information and belong in the authorized
environment, along with inputs, checkpoints and per-case outputs. The code
does not establish that a remote endpoint meets a particular data agreement.

Dataset sources: [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/),
[MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/).
Field semantics: [lab events](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html),
[microbiology](https://mimic.mit.edu/docs/iv/modules/hosp/microbiologyevents.html).

## Repository layout

```text
src/mimic/       extraction, timeline construction, note section parsing
src/tasks/       B1/B3 tasks, splits, rendering and temporal policy
src/runners/     single-model runners, validation and checkpoint handling
src/eval/        scoring functions
src/traps/       experimental perturbation generation and scoring
src/llm/         provider adapters, client, concurrency
tests/          offline synthetic tests
scripts/        release checks and controlled-environment analysis helpers
docs/           limitations and B3 migration
```

## License, citation and contributions

Code: [Apache License 2.0](LICENSE). Copyright 2026 the HLCD Benchmark authors.
See [NOTICE](NOTICE). Cite the software using [CITATION.cff](CITATION.cff),
and cite the databases separately when using them. Repository and archive links
can be added after publication; no publication DOI or journal acceptance is
claimed by this source release.

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
Changes: [CHANGELOG.md](CHANGELOG.md).
