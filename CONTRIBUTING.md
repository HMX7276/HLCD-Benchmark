# Contributing

Use synthetic examples in bug reports, tests and pull requests. Do not upload
credentials, records, patient identifiers, transcripts or reviewer materials.

Create a branch, describe the problem and expected behavior, and run:

```text
python -m unittest discover -s tests -v
python scripts/check_public_release.py
```

Method changes should document how labels, inputs, denominators, failure rules
and prior results are affected. Add focused regression tests. Preserve historical
results rather than silently replacing them. Provider adapters need validation
against the actual endpoint/model version; offline tests alone do not prove API
compatibility.

By submitting a contribution, you agree to license that contribution under the
repository's Apache-2.0 license. Supply only code you have the right to contribute.
