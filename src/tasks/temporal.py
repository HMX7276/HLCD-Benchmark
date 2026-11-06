"""Conservative B3 availability policy; naive timestamps follow MIMIC time.

Day X covers [admission + X days, admission + (X+1) days). Final result
content is never backdated to collection time. This is not a reconstruction
of interim results or an availability audit of every other source table.
"""
from datetime import datetime, timedelta

TIMELINE_SCHEMA_VERSION = 3
B3_INPUT_POLICY = "b3-availability-v1"
RESULT_KINDS = frozenset({"lab", "microbio", "radiology"})


def timestamp(value):
    if value is None or str(value).strip().lower() in {"", "nat", "nan", "none", "<na>"}:
        return None
    try:
        ts = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    # MIMIC timestamps are timezone-naive. Do not silently convert mixed clocks.
    return ts if ts.tzinfo is None else None


def result_availability(event_time, storetime=None, storedate=None):
    """Persist availability independently of event day; date-only upper bound.

    Missing exact time can use storedate only when the caller supplies it
    (microbiology). Malformed exact times never fall back to a date.
    """
    raw = str(storetime).strip().lower()
    missing = storetime is None or raw in {"", "nat", "nan", "none", "<na>"}
    event = timestamp(event_time)
    at = timestamp(storedate) if missing else timestamp(storetime)
    precision = "date_upper_bound" if missing else "exact"
    field = "storedate" if missing else "storetime"
    if at is None or event is None:
        return {"status": "unknown_or_invalid", "at": None, "precision": precision,
                "source_field": field}
    if missing:
        at = at.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    # For date-only availability, the whole reported date precedes the event
    # if even its exclusive upper bound is not later than the event time.
    if at < event or (missing and at == event):
        return {"status": "before_event", "at": None, "precision": precision,
                "source_field": field}
    return {"status": "known", "at": at.isoformat(), "precision": precision,
            "source_field": field}


def prediction_cutoff(doc, max_day):
    if isinstance(max_day, bool) or not isinstance(max_day, int) or max_day < 0:
        raise ValueError("B3 max_day must be a nonnegative integer")
    admit = timestamp(doc.get("meta", {}).get("admit_time"))
    if admit is None:
        raise ValueError("B3 requires a valid timezone-naive meta.admit_time")
    return admit + timedelta(days=max_day + 1)


def result_visible(record, cutoff):
    """Return (visible, reason), including validation of serialized metadata."""
    info = record.get("availability") or {}
    at = timestamp(info.get("at"))
    event = timestamp(record.get("anchored_time"))
    precision = info.get("precision")
    if (info.get("status") != "known" or at is None or event is None
            or precision not in {"exact", "date_upper_bound"}):
        return False, "unknown_or_invalid_availability"
    if at < event or (precision == "date_upper_bound" and at == event):
        return False, "availability_before_event"
    visible = at <= cutoff if precision == "date_upper_bound" else at < cutoff
    return visible, "visible" if visible else "available_after_cutoff"
