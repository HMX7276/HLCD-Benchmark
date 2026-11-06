# -*- coding: utf-8 -*-
"""MIMIC 文书段切分（discharge summary / radiology 报告，英文标签）。"""
import re

_DS_HEADERS = [
    "Chief Complaint", "Major Surgical or Invasive Procedure",
    "History of Present Illness", "Past Medical History",
    "Medications on Admission", "Allergies", "Social History",
    "Family History", "Review of Systems", "Physical Exam",
    "Pertinent Results", "Brief Hospital Course", "Hospital Course",
    "Hospitalizations", "Discharge Diagnosis", "Discharge Diagnoses",
    "Discharge Medication", "Discharge Medications", "Discharge Disposition",
    "Discharge Condition", "Discharge Instructions",
    "Followup Instructions", "Follow-up Instructions",
]
_DS_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(h) for h in _DS_HEADERS) +
    r")\s*:\s*", re.I | re.M)

_RAD_HEADERS = ["EXAMINATION", "INDICATION", "TECHNIQUE", "FINDINGS",
                "IMPRESSION", "ADDENDUM", "COMPARISON", "RECOMMENDATION"]
_RAD_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(h) for h in _RAD_HEADERS) +
    r")\s*:\s*", re.M)


def split_sections(text: str, pattern: re.Pattern) -> dict[str, str]:
    marks = list(pattern.finditer(text))
    out = {}
    for j, m in enumerate(marks):
        end = marks[j + 1].start() if j + 1 < len(marks) else len(text)
        header = m.group(0).strip().rstrip(":").strip()
        out[_norm_key(header)] = text[m.end():end].strip()
    return out


def _norm_key(h: str) -> str:
    h = h.lower().replace("diagnoses", "diagnosis").replace(
        "medications", "medication").replace("medication on admission",
                                             "medication_admission")
    for canon in ("hospital course", "discharge diagnosis",
                  "discharge medication", "discharge instructions",
                  "followup instructions"):
        if canon in h:
            return canon
    return h


def parse_discharge(text: str) -> dict:
    """出院小结 → 段字典（键为规范小写名）。"""
    return split_sections(text, _DS_RE)


def parse_radiology(text: str) -> dict:
    """放射报告 → {examination/indication/findings/impression/...}。"""
    return split_sections(text, _RAD_RE)
