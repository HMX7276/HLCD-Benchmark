# Known limitations

The implementation addresses the reproduced software defects; passing tests is
not clinical validation or proof of historical benchmark accuracy.

- B1 references from withheld discharge summaries may not be recoverable from
  the visible input. Expert-validated visible references are not included.
- B1 assertion matching is a conservative English rule system. Complex negation,
  mixed clauses, abbreviations and coreference need human validation. The score
  remains assertion-aware lexical recall, not verified clinical factual accuracy.
- B3 input availability is enforced for lab/microbiology/radiology results and
  future stop dates. Other sources, including daily derived measurements, coded
  procedures and retrospective medication updates, are not fully audited.
- Missing in-hospital death timestamps exclude cases from B3 and are counted.
  Exclusions and valid-response-only metrics can introduce selection differences;
  compare denominators/failure coverage alongside scores.
- Event definitions remain operational proxies: antibiotic_start is the first
  matching medication record in the admission, not every new drug/course; the
  current list includes some antifungals. major_procedure uses coded procedures,
  and new_key_exam is a count/imaging rule. Clinical validity requires review.
- Perturbation targets and scores are explicit, but recognition/adoption remain
  heuristic assessments. An omitted target is not credited as detection. Some
  cases remain unscorable; paired clean warnings are not automatically false
  alarms. Expert validation is required before claims of clinical detection.
- The code does not include actual cohort records, complete historical run
  settings, expert-review evidence or the SQL creating derived vitals/GCS tables.
  Examples are not historical experiment settings.
- Provider model IDs, API behavior and data-handling agreements require local
  verification. Byte-based context budgeting is conservative, not a provider
  tokenizer. Provider-side revisions cannot be inferred from an unchanged URL.

See [v0.2 migration](MIGRATION_V0.2.md) for changed schemas, labels, scoring and
failure denominators. No actual model experiments or patient-data analyses were
rerun as part of these source fixes. No journal acceptance is claimed.
