"""Source hashing and B3 timeline schema validation."""
import hashlib
import json
from pathlib import Path

from src.tasks.temporal import B3_INPUT_POLICY, TIMELINE_SCHEMA_VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def source_digest():
    root = Path(__file__).resolve().parents[1]
    return digest({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob("*.py"))})


def validate_timeline(doc):
    if doc.get("timeline_schema_version") != TIMELINE_SCHEMA_VERSION:
        raise ValueError("B3 requires rebuilt schema-v3 timelines. Re-export result times "
                         "and rebuild timelines in a new work directory; legacy timelines "
                         "cannot be resumed as corrected inputs.")

