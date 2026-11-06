# Migration to v0.2.0

This release contains only single-model inference. Consensus/window voting code,
configuration and tests have been removed. The public entry points are:

```text
python -m src.runners.run_b1 --n 20 --out B1_RESULT.json --ckpt B1_CHECKPOINT.jsonl
python -m src.runners.run_b3 --n 20 --X 1 --out B3_RESULT.json --ckpt B3_CHECKPOINT.jsonl
python -m src.traps.evaluate --out TRAP_RESULT.json --ckpt TRAP_CHECKPOINT.jsonl
```

There is no `--mode` option. Configure one `BASE_MODEL`/`BASE_URL`/`BASE_PROVIDER`/
`BASE_KEY` slot or pass the model/provider CLI overrides. Run each model with
its own output/checkpoint path. Actual data/model work belongs in the authorized
environment; offline tests do not require either.

## Required regeneration

- Rebuild timeline **schema 3** from the frozen original cohort exports. Preserve
  `admittime`, `dischtime`, `deathtime`, and result availability timestamps.
  Do not substitute rounded length of stay for exact timestamps. Missing or
  invalid in-hospital death time makes that B3 case unlabelable; it is excluded
  and counted. A patient who died before the prediction cutoff is not eligible.
- B3 uses `b3-exact-events-v2` labels and `b3-availability-v1` inputs. Audit the
  affected case list; label-only changes may permit rescoring preserved answers,
  while changed inputs or newly eligible cases require inference.
- Rebuild perturbations with schema **2**, paired clean/tampered inputs and
  explicit target metadata. Old targetless traps and boolean-only checkpoints
  are rejected. The six operation types include `temporal_order_swap`, replacing
  the misleading old causal label. `polarity_flip` is retained as a compatibility
  type name but means **flag removal**, not an inversion of physiological truth.
- Old checkpoints are incompatible. All runners bind successes to task, code,
  model route/settings and the input/reference context. Foreign sample rows,
  changed inputs, malformed predictions and torn files are rejected before
  inference. Failed cases are saved in final output but retried on resume.

## New metric meanings and denominators

`b1-assertion-lexical-v2` separates affirmative, negative, uncertain, historical
and family-history assertions before lexical matching. This is a conservative
English heuristic, not a validated general clinical assertion parser. B1
metrics use valid responses with nonempty references and report failures and
the total/valid response counts separately. Reference observability remains
an independent issue.

B3 requires every event and accepts only integer 0/1 or exact strings "0"/"1".
Empty/missing/invalid outputs are failures, not all-negative predictions.
Metrics use **valid responses only**, with `n_total`, `n_valid`, `fails` and
eligibility counts reported. Zero valid predictions produces null metrics.
This differs from historical failure-as-zero scoring; comparing models requires
reviewing coverage/failure rates together with conditional performance.

`traps-target-aware-v2` reports four states: `explicit_recognition`,
`incorrect_adoption`, `not_mentioned`, `unscorable`. The latter two never count
as interception success. Matching uses response values and target entity,
numeric value/time, not JSON key names or unrelated alarms. For value/flag
targets with a known day, an unspecified or different response day is unresolved.
Both clean and perturbed inputs use the same summarization prompt, including
optional target-specific warnings. Per-type counts and valid-response rates are
reported separately; paired-valid counts and clean targeted-warning rates are
included. Clean warnings are not automatically false positives: genuine input
uncertainty requires human review. These heuristic states still need expert
validation before being interpreted as clinical error-detection ability.

The output now has `metrics` and per-response `predictions` rows with explicit
status, prediction/error and provenance. Perturbation rows preserve model
summaries and paired assessments for later local rescoring. Treat all real
outputs/checkpoints as controlled research material.

## Runtime and compatibility

Native Ollama, Anthropic and OpenAI envelopes are checked separately. Completion
requests are capped by configured generation budget and a conservative UTF-8
byte estimate of remaining context. No request is made when the estimate leaves
no room. The estimate is not an exact provider tokenizer; configure context
correctly and inspect provider usage. Internal retries cannot raise the request
above its original ceiling.

Extraction now rejects nonpositive cohort size or an insufficient candidate
pool with a clear message. Reduce the requested size deliberately or correct
eligibility settings; it no longer attempts sampling from an exhausted pool.

Historical tables, manuscript results and old ZIPs have not been rewritten.
Archive new analyses separately and update the manuscript only from completed
authorized experiments and their audit records.
