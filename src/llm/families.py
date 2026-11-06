# -*- coding: utf-8 -*-
"""Model-family registry: name matching, endpoints, keys, thinking adapters.

Cloud families are reached through their OpenAI-compatible
/chat/completions endpoint. Endpoint defaults and thinking-parameter
shapes are best-effort conventions: only families with a key at hand have
been live-verified; the rest are table-driven guesses that may need a
one-line fix once their key is available (see `_VERIFIED`).

Model spec syntax accepted everywhere a model is expected:
    "model"                 plain id (auto family/provider routing)
    "provider::model"       force transport (vllm / ollama / openai)
    "model@LEVEL"           per-model thinking level (off|low|medium|high)
"""
import os
import re

THINK_LEVELS = ("auto", "off", "min", "low", "medium", "high", "max")
_LEVEL_RE = re.compile(r"^(?P<model>.+?)@(?P<level>off|min|low|medium|high|max)$")

# families live-verified against a real key so far
_VERIFIED = ("deepseek",)

# match: case-insensitive substrings against BARE model ids only -- a "/"
# (HF-style repo path) or ":" (ollama tag) marks a local deployment and
# never matches a cloud family.
FAMILIES = {
    "deepseek": {"match": ("deepseek",),
                 "url": "https://api.deepseek.com/v1",
                 "url_env": "DEEPSEEK_BASE_URL", "key_env": "DEEPSEEK_API_KEY",
                 "thinking": "deepseek"},
    "qwen": {"match": ("qwen",),
             "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "url_env": "DASHSCOPE_BASE_URL", "key_env": "DASHSCOPE_API_KEY",
             "thinking": "qwen_cloud"},
    "glm": {"match": ("glm",),
            "url": "https://open.bigmodel.cn/api/paas/v4",
            "url_env": "ZHIPU_BASE_URL", "key_env": "ZHIPU_API_KEY",
            "thinking": "glm"},
    "kimi": {"match": ("kimi", "moonshot"),
             "url": "https://api.moonshot.cn/v1",
             "url_env": "MOONSHOT_BASE_URL", "key_env": "MOONSHOT_API_KEY",
             "thinking": "kimi"},
    "gpt": {"match": ("gpt-",),
            "url": "https://api.openai.com/v1",
            "url_env": "OPENAI_BASE_URL", "key_env": "OPENAI_API_KEY",
            "thinking": "gpt"},
    "claude": {"match": ("claude",),
               "url": "https://api.anthropic.com/v1",
               "url_env": "ANTHROPIC_BASE_URL", "key_env": "ANTHROPIC_API_KEY",
               "thinking": "claude"},
    "gemini": {"match": ("gemini",),
               "url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "url_env": "GEMINI_BASE_URL", "key_env": "GEMINI_API_KEY",
               "thinking": "gemini"},
    "grok": {"match": ("grok",),
             "url": "https://api.x.ai/v1",
             "url_env": "XAI_BASE_URL", "key_env": "XAI_API_KEY",
             "thinking": "grok"},
}

# token budgets for adapters that support thinking granularity
THINK_BUDGET = {"low": 2048, "medium": 8192, "high": 16384}


def parse_spec(spec: str) -> tuple[str, str | None]:
    """Split "model@level" -> (model, level|None); reject bad levels."""
    m = _LEVEL_RE.match(spec.strip())
    if m:
        return m.group("model"), m.group("level")
    if "@" in spec:
        level = spec.rsplit("@", 1)[1]
        raise ValueError(
            f"bad thinking level {level!r} in {spec!r}: "
            f"use @off / @low / @medium / @high")
    return spec.strip(), None


def match_family(model: str) -> str | None:
    """Cloud family for a model id, regardless of how it is served.

    Matching runs on the basename so a vLLM repo path ("Qwen/Qwen3.8-72B")
    is recognised the same as its cloud id; an ollama tag ("qwen3.5:9b")
    is NOT a family match -- its transport is ollama and nothing else
    is needed. Family decides thinking semantics, transport decides the
    wire format.
    """
    if ":" in model:
        return None  # ollama tag: transport is known, family not needed
    bare = model.rsplit("/", 1)[-1].lower()
    if not bare:
        return None
    for name, fam in FAMILIES.items():
        if any(p in bare for p in fam["match"]):
            return name
    return None


def family_endpoint(family: str) -> tuple[str, str]:
    """(chat_url, api_key) for a cloud family.

    The family's own key env is the only source; no LLM_API_KEY fallback.
    """
    fam = FAMILIES[family]
    base = (os.getenv(fam["url_env"], fam["url"])).rstrip("/")
    if family == "claude":
        # native /v1/messages, not the OpenAI-compatible shape
        url = base if base.endswith("/messages") else (
            base + "/messages" if base.endswith("/v1")
            else base + "/v1/messages")
        return url, os.getenv(fam["key_env"], "")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    key = os.getenv(fam["key_env"], "")
    return url, key


# token budgets for adapters that support thinking granularity
THINK_BUDGET = {"min": 1024, "low": 2048, "medium": 8192, "high": 16384,
                "max": 32768}


def thinking_payload(kind: str, level: str, max_tokens: int | None = None
                     ) -> dict:
    """Chat-completions payload fragment expressing the thinking level.

    Adapter shapes follow each family's official docs (verified 2026-09):
      deepseek  thinking:{type} + reasoning_effort low|high|max
                (docs map: low->low, medium->high, high->high, max->max)
      qwen_cloud  enable_thinking + thinking_budget 1..32768 (no effort tiers)
      glm       thinking:{type} + reasoning_effort low|high|max (GLM-5.2+)
      kimi      reasoning_effort supports ONLY "max" (K3 always thinks)
      gpt       reasoning_effort minimal|low|medium|high (default medium)
      gemini    reasoning_effort minimal|low|medium|high ("none" only 2.5)
      grok      reasoning_effort low|medium|high (default high, no off)
      claude    native thinking:{type:enabled, budget_tokens} -- the budget
                must stay below max_tokens, so it is clamped to it
    "auto" sends nothing anywhere (model default); "off" on families that
    cannot disable thinking also sends nothing.
    """
    if level == "auto":
        return {}

    if kind == "deepseek":
        if level == "off":
            return {"thinking": {"type": "disabled"}}
        return {"reasoning_effort":
                {"min": "low", "low": "low", "medium": "high",
                 "high": "high", "max": "max"}[level]}
    if kind == "qwen_cloud":
        if level == "off":
            return {"enable_thinking": False}
        return {"enable_thinking": True,
                "thinking_budget": min(THINK_BUDGET[level], 32768)}
    if kind == "glm":
        if level == "off":
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "enabled"},
                "reasoning_effort":
                {"min": "low", "low": "low", "medium": "high",
                 "high": "high", "max": "max"}[level]}
    if kind == "kimi":  # K3: thinking always on, effort only supports max
        return {"reasoning_effort": "max"} if level == "max" else {}
    if kind == "gpt":
        return {"reasoning_effort":
                {"off": "minimal", "min": "minimal", "low": "low",
                 "medium": "medium", "high": "high", "max": "high"}[level]}
    if kind == "gemini":
        if level == "off":
            return {}
        return {"reasoning_effort":
                {"min": "minimal", "low": "low", "medium": "medium",
                 "high": "high", "max": "high"}[level]}
    if kind == "grok":
        if level == "off":
            return {}
        return {"reasoning_effort":
                {"min": "low", "low": "low", "medium": "medium",
                 "high": "high", "max": "high"}[level]}
    if kind == "claude":
        # native /v1/messages: budget must fit inside max_tokens
        budget = THINK_BUDGET[level]
        if max_tokens and max_tokens - 512 < budget:
            headroom = max_tokens - 512
            if headroom < 1024:  # no room for even the minimum budget
                return {"thinking": {"type": "enabled"}}
            budget = headroom
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    raise ValueError(f"unknown thinking adapter {kind!r}")


def vllm_thinking(level: str) -> dict:
    """chat_template_kwargs for self-hosted vLLM (Qwen3-style template).

    "auto" keeps thinking ON (the template default): set @off /
    $LLM_THINKING=off to disable; tiers add a thinking_budget.
    """
    if level == "off":
        return {"enable_thinking": False}
    if level == "auto":
        return {"enable_thinking": True}
    return {"enable_thinking": True, "thinking_budget": THINK_BUDGET[level]}


def parse_model_list(specs: str, limit: int = 5) -> list[str]:
    """Comma list -> validated spec list (each entry keeps its @level)."""
    out = [s.strip() for s in (specs or "").split(",") if s.strip()]
    if not out:
        raise ValueError("model list is empty")
    for s in out:
        parse_spec(s)  # raises on a bad @level
    if len(out) > limit:
        raise ValueError(f"{len(out)} models exceeds the cap of {limit}: {out}")
    return out


def parse_slots(slots, limit: int = 5) -> list[dict]:
    """config._slots() output -> validated [{model, url, provider}]; cap limit."""
    out = [{"model": m, "url_override": u, "provider": p}
           for m, u, p in slots if m]
    if not out:
        raise ValueError("model list is empty")
    for s in out:
        parse_spec(s["model"])  # raises on a bad @level
    if len(out) > limit:
        raise ValueError(f"{len(out)} models exceeds the cap of {limit}: "
                         f"{[s['model'] for s in out]}")
    return out
