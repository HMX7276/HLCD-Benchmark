# -*- coding: utf-8 -*-
"""Global config: everything user-tunable lives in `.env` (see .env.example).

Precedence: CLI args > environment variables / .env > defaults here.
Paths are derived once here; other modules must import from this module
instead of hardcoding (single source of truth).
"""
import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(PROJECT_ROOT, ".env")):
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
else:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env.example"))


def _int(name, default):
    return int(os.getenv(name, str(default)))


def _float(name, default):
    return float(os.getenv(name, str(default)))


def _csv(name, default):
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def _base_slot():
    """Singular base slot: BASE_URL / BASE_MODEL / BASE_PROVIDER / BASE_KEY."""
    m = os.getenv("BASE_MODEL", "").strip()
    if not m:
        return None
    return (m,
            os.getenv("BASE_URL", "").strip().rstrip("/") or None,
            os.getenv("BASE_PROVIDER", "").strip().lower() or None,
            os.getenv("BASE_KEY", "").strip() or None)


# ---------------- LLM provider ----------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # fallback when a slot omits provider
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0)
# vllm 专用：打破 GGUF 部署的退化重复循环（1.0 = 不发送）
LLM_REPETITION_PENALTY = _float("LLM_REPETITION_PENALTY", 1.0)
# Slot-style config, no legacy lists:
#   base: BASE_URL / BASE_MODEL / BASE_PROVIDER / BASE_KEY
# A slot with only a URL (no model) is ignored; a slot with only a model
# gets its family default endpoint (or an error if the family is unknown).
BASE_SLOT = _base_slot()
# global thinking level (auto|off|min|low|medium|high|max); per-model "@level" wins
LLM_THINKING = os.getenv("LLM_THINKING", "auto").strip().lower()
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 2000)
LLM_RETRIES = _int("LLM_RETRIES", 3)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 600)
LLM_MAX_TOKEN_BUDGET = _int("LLM_MAX_TOKEN_BUDGET", 16000)
LLM_CONCURRENCY = max(1, _int("LLM_CONCURRENCY", 1))  # 1=串行；跑全量建议 4–16

# ---------------- MIMIC PostgreSQL (read-only) ----------------
PG = {"host": os.getenv("PG_HOST", "localhost"),
      "port": os.getenv("PG_PORT", "5432"),
      "user": os.getenv("PG_USER", "postgres"),
      "password": os.getenv("PG_PASSWORD", "")}
PG_DB_MIMICIV = os.getenv("PG_DB_MIMICIV", "mimiciv")
PG_DB_MIMICNOTE = os.getenv("PG_DB_MIMICNOTE", "mimicnote")

# ---------------- workspace & derived paths ----------------
HLCD_WORK = os.path.expanduser(os.getenv("HLCD_WORK", "~/hlcd_work"))
MIMIC_RAW = os.path.join(HLCD_WORK, "mimic")            # exported tables
STRUCTURED_DIR = os.path.join(HLCD_WORK, "structured")  # timeline parent
TIMELINE_VERSION = os.getenv("TIMELINE_VERSION", "mimic_v1.0")
TIMELINE_DIR = os.path.join(STRUCTURED_DIR, TIMELINE_VERSION)
TASKS_DIR = os.path.join(HLCD_WORK, "mimic_tasks")
B1_DATASET = os.path.join(TASKS_DIR, "b1_dataset.jsonl")
SPLITS_FILE = os.path.join(TASKS_DIR, "splits.json")
TRAPS_FILE = os.path.join(TASKS_DIR, "traps.jsonl")
RUNS_DIR = os.path.join(HLCD_WORK, "mimic_runs")
REPORTS_DIR = os.path.join(HLCD_WORK, "reports")
DOCS_DIR = os.path.join(HLCD_WORK, "checks")

# ---------------- cohort extraction ----------------
COHORT_SIZE = _int("COHORT_SIZE", 2000)
COHORT_SEED = _int("COHORT_SEED", 42)
MIN_LOS_DAYS = _float("MIN_LOS_DAYS", 2.0)  # must be >= max(B3 X)+1 to support B3

# ---------------- task construction ----------------
SPLITS_SEED = _int("SPLITS_SEED", 42)
SPLIT_RATIO = os.getenv("SPLIT_RATIO", "5:1:3")  # train:val:test parts
B3_X_VALUES = [int(x) for x in _csv("B3_X_VALUES", "1,3,5")]
B3_NEW_EXAM_LABS = _int("B3_NEW_EXAM_LABS", 5)
TRAPS_N_PER_TYPE = _int("TRAPS_N_PER_TYPE", 5)
TRAPS_SEED = _int("TRAPS_SEED", 13)

EVAL_SEED = _int("EVAL_SEED", 7)

# ---------------- rendering & e2e ----------------
RENDER_MAX_ABNORMAL_LABS = _int("RENDER_MAX_ABNORMAL_LABS", 20)
E2E_ANCHORED_MIN = _float("E2E_ANCHORED_MIN", 0.95)
