# Changelog

## 0.2.0 — single-model correctness revision (prepared 2026-11-05)

- Removed consensus/window voting implementation, configuration and CLI modes.
  Single-model entry points now live in `src/runners`.
- Strict binary/summary validation, explicit failure status, valid-response
  denominators and task/input/model/reference-bound checkpoints for all runners.
- Exact discharge/death labels and eligibility (timeline schema 3); unknown
  in-hospital death times are excluded and counted.
- Target-aware paired perturbations and four descriptive outcome states;
  event-date swaps replace the former causal label. No legacy interception rate.
- Assertion-aware B1 lexical scoring; native Ollama envelope handling; bounded
  completion requests/retries; clear insufficient-cohort errors.
- Updated offline regression/integration tests, reports and migration notes.
- Scoped the raw-data ignore rule to `/mimic/` so `src/mimic` is tracked.
- Normalized source line endings to LF for consistent Git/archive checksums.
- No historical results or real patient datasets changed. These are software
  fixes and new heuristic definitions, not a completed validation experiment.

## 0.1.0 — archived initial research source package

Contained B3 stored-availability filtering, future-stop masking and source-only
packaging. That ZIP is retained for provenance, not the current release.
