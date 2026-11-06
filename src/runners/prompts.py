"""Single-model structured prompts."""
from src import config

B1_SYSTEM = """You are a clinical records summarization assistant. Based on the
given inpatient timeline, produce a structured JSON summary.
Use only information present in the timeline. Do not fabricate.
Output JSON only (no other text):
{
 "diagnoses": [{"name": "diagnosis", "day": first-seen day or null}],
 "surgeries": [{"name": "procedure name", "day": day or null}],
 "medications": [{"name": "drug", "day": start day or null}],
 "key_abnormal_labs": [{"name": "lab test", "value": "value and direction", "day": null}],
 "key_imaging_findings": [{"name": "imaging conclusion", "day": null}],
 "allergy": "allergy text, empty string if none",
 "outcome": "one-sentence outcome",
 "warnings": [{"target": "specific entity or missing input", "reason": "explicit concern; use [] if none"}]
}
Notes: day counts are relative to admission (admission day = Day 0). Prefer
canonical procedure names (e.g., "TURP", "central line placement")."""

B1_USER = """Inpatient timeline:
{timeline}

Output the JSON summary."""

# keep the "N+ distinct lab tests" wording in sync with $B3_NEW_EXAM_LABS
_B3_DEFS = """Definitions: "discharge" = discharged on Day {X1}; "icu_transfer" = transferred
into an ICU on Day {X1}; "extubation" = endotracheal tube removed; "new_key_exam"
= 5+ distinct lab tests or any imaging on Day {X1}."""
_B3_DEFS = _B3_DEFS.replace("5+ distinct lab tests",
                            f"{config.B3_NEW_EXAM_LABS}+ distinct lab tests")

B3_SYSTEM = """You are a clinical trajectory prediction assistant. Based on the
patient timeline from Day 0 to Day {X}, predict which key events happen on
Day {X1}. Output JSON:
{{
 "predictions": {{
   "discharge": 0 or 1,
   "death": 0 or 1,
   "icu_transfer": 0 or 1,
   "antibiotic_start": 0 or 1,
   "antibiotic_stop": 0 or 1,
   "extubation": 0 or 1,
   "transfusion": 0 or 1,
   "major_procedure": 0 or 1,
   "new_key_exam": 0 or 1
 }},
 "reasoning": "one-sentence rationale"
}}
""" + _B3_DEFS + """ Output JSON only."""

B3_USER = """Patient timeline (Day 0-{X}):
{timeline}

Predict Day {X1} events. Output JSON."""

