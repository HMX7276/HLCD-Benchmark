# -*- coding: utf-8 -*-
"""Unified LLM client. Provider/model/URL/key all come from `.env`
(see .env.example); CLI args override. Transports:
  vllm   - OpenAI-compatible server you run (thinking via chat_template_kwargs)
  ollama - Ollama /api/chat (thinking via the `think` flag)
  openai - any OpenAI-compatible cloud API
Model ids are resolved through src/llm/families.py: bare cloud names
("deepseek-v4-flash", "gpt-5", "claude-sonnet-5", "glm-5.3", ...) pick
their family endpoint+key automatically; "Qwen/..." -> vllm and "tag:9b"
-> ollama remain local heuristics; "provider::model" forces a transport.
Thinking: global $LLM_THINKING plus per-model "@off|low|medium|high".
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from src import config
from src.llm import families
from src.llm.budget import completion_budget

TRANSCRIPT = os.path.join(config.RUNS_DIR, "transcripts.jsonl")
_TLOCK = threading.Lock()  # concurrent workers append to the same jsonl


class LLMError(RuntimeError):
    pass


class _EnvelopeError(LLMError):
    """Malformed HTTP envelope from a gateway (transient overload artifact);
    retryable, unlike a budget-exhaustion LLMError which must propagate."""


# Start-spacing rate gate for shared free-tier endpoints. OpenRouter free
# models cap at 20 requests/min (sliding 60s window): spacing request STARTS
# ~3.05s apart keeps throughput just under the cap while LLM_CONCURRENCY
# requests stay in flight. Only gates the send moment, not responses.
# Optional per-key spacing via $LLM_KEY_MIN_INTERVAL="<key_prefix>=<sec>,…":
# when one upstream slot degrades to ~100-150s/service and a fixed wall-clock
# kill (CSTCloud 422) waits at ~180s, spacing that key's request starts
# prevents queue pile-up → mass 422 retry storms.
_RL_LOCK = threading.Lock()
_RL_NEXT = 0.0  # monotonic time of the earliest allowed next start
_RL_KEY_NEXT: dict = {}


def _hot_file_lines(name: str) -> list[str]:
    """result/<name> hot switch: rows read per call (tune without restart)."""
    try:
        with open(os.path.join(config.PROJECT_ROOT, "result", name),
                  encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return []


def _env_key_val(env_name: str, key: str | None) -> str | None:
    """Value from env specs like LLM_KEY_TIMEOUT="<prefix>=<sec>,…" for the
    matching key prefix; None when the key is not covered."""
    if not key:
        return None
    for part in os.getenv(env_name, "").split(","):
        if "=" not in part:
            continue
        prefix, _, val = part.partition("=")
        if key.startswith(prefix.strip()):
            return val.strip()
    return None


def _hot_float(name: str) -> float | None:
    """Single float from result/<name> (hot-tunable, read per call)."""
    for ln in _hot_file_lines(name):
        try:
            return float(ln)
        except ValueError:
            continue
    return None


def _key_disabled(key: str | None) -> bool:
    """result/_key_disabled.txt lists key prefixes to skip instantly
    (degraded upstream): generate() raises LLMError without any HTTP call,
    so the caller can record a failed request."""
    if not key:
        return False
    return any(key.startswith(p) for p in _hot_file_lines("_key_disabled.txt"))


def _key_timeout(key: str | None) -> float:
    """Per-key socket timeout; the hot file only affects keys covered by the
    env spec (healthy keys are never throttled by a stale hot file)."""
    val = _env_key_val("LLM_KEY_TIMEOUT", key)
    if val is None:
        return 0.0
    hot = _hot_float("_key_timeout.txt")
    sec = hot if hot is not None and hot > 0 else float(val)
    return max(0.0, sec)


def _key_min_interval(key: str | None) -> float:
    if not key:
        return 0.0
    val = _env_key_val("LLM_KEY_MIN_INTERVAL", key)
    if val is None:
        return 0.0
    hot = _hot_float("_key_rate_interval.txt")
    sec = hot if hot is not None and hot >= 0 else float(val)
    return max(0.0, sec)


def _rate_gate(url: str, min_interval: float = 3.05, key: str | None = None):
    global _RL_NEXT
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if "openrouter.ai" not in host:
        key_interval = _key_min_interval(key)
        if key_interval <= 0:
            return
        with _RL_LOCK:
            now = time.monotonic()
            wait = max(0.0, _RL_KEY_NEXT.get(key, 0.0) - now)
            _RL_KEY_NEXT[key] = max(now, _RL_KEY_NEXT.get(key, 0.0)) + key_interval
        if wait > 0:
            time.sleep(wait)
        return
    with _RL_LOCK:
        now = time.monotonic()
        wait = max(0.0, _RL_NEXT - now)
        _RL_NEXT = max(now, _RL_NEXT) + min_interval
    if wait > 0:
        time.sleep(wait)


def resolve(spec: str | None = None, provider: str | None = None,
            url_override: str | None = None,
            key_override: str | None = None) -> dict:
    """Full resolution of a model spec -> routing + thinking level.

    Returns {provider, model, family, url, key, thinking, display}.
    Slot-style config (config._base_slot) passes url_override,
    provider and key_override explicitly; anything left empty falls back to
    family defaults or $LLM_PROVIDER (default auto).
    """
    model, level = families.parse_spec(spec or "")
    thinking = level or config.LLM_THINKING
    if thinking not in families.THINK_LEVELS:
        raise ValueError(f"bad LLM_THINKING {thinking!r}: "
                         f"auto|off|min|low|medium|high|max")
    forced = None
    if "::" in model:
        forced, model = model.split("::", 1)
    fam = families.match_family(model)
    prov = (forced or provider or config.LLM_PROVIDER or "auto").lower()
    if prov == "auto":
        prov = "ollama" if (":" in model and "/" not in model) else "vllm"

    key = key_override
    if url_override:
        base = url_override.rstrip("/")
        if prov == "ollama":
            url = base + "/api/chat"
        elif prov in ("claude", "anthropic"):
            # Anthropic-style native endpoint: .../v1/messages (an explicit
            # base may be the host root, .../v1, or the full messages URL)
            url = base if base.endswith("/messages") else (
                base + "/messages" if base.endswith("/v1")
                else base + "/v1/messages")
        else:
            url = (base if base.endswith("/chat/completions")
                   else base + "/chat/completions")
        return {"provider": prov, "model": model, "family": fam, "url": url,
                "key": key, "thinking": thinking,
                "display": f"{model}@{thinking}" if level else model}
    if fam:
        url, fam_key = families.family_endpoint(fam)
        return {"provider": "openai", "model": model, "family": fam,
                "url": url, "key": key or fam_key, "thinking": thinking,
                "display": f"{model}@{thinking}" if level else model}
    raise LLMError(f"cannot resolve {model!r}: set URL (and optionally "
                   f"PROVIDER/KEY) in .env slot-style "
                   f"(<LINE>_URL<i>/<LINE>_PROVIDER<i>/<LINE>_KEY<i>), "
                   f"or use provider::model")


def route(model: str | None = None, provider: str | None = None,
          url_override: str | None = None,
          key_override: str | None = None) -> tuple[str, str]:
    """Back-compat thin wrapper: resolve() -> (provider, model)."""
    r = resolve(model, provider, url_override, key_override)
    return r["provider"], r["model"]


def generate(model: str | None = None, messages: list[dict] | None = None,
             max_tokens: int = None, temperature: float = None,
             endpoint: str | None = None, url_override: str | None = None,
             key_override: str | None = None) -> dict:
    """Returns {text, usage, latency, model, provider, family, thinking};
    raises LLMError."""
    if messages is None:
        raise ValueError("messages is required")
    max_tokens = completion_budget(messages, max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS)
    request_ceiling = max_tokens
    r = resolve(model, endpoint, url_override, key_override)
    if _key_disabled(r["key"]):
        raise LLMError(f"{r['model']}: disabled by _key_disabled hot switch")
    provider, m = r["provider"], r["model"]
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    # 浏览器 UA：部分云网关（Cloudflare bot 检测）会以 403/1010
    # 拦截 Python-urllib 默认 UA
    headers = {"Content-Type": "application/json",
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/126.0.0.0 Safari/537.36"}
    if provider not in ("ollama", "claude", "anthropic") and r["key"]:
        headers["Authorization"] = f"Bearer {r['key']}"
    if provider in ("claude", "anthropic") and r["key"]:
        headers["x-api-key"] = r["key"]
        headers["anthropic-version"] = "2023-06-01"

    if provider == "ollama":
        payload = {"model": m, "messages": messages, "stream": False,
                   "options": {"temperature": temperature,
                               "num_predict": max_tokens}}
        if r["thinking"] != "auto":
            payload["think"] = r["thinking"] != "off"
    elif provider in ("claude", "anthropic"):
        # Anthropic-style: system extracted, max_tokens mandatory; the
        # thinking budget (if any) must stay below max_tokens
        sys_msg = next((x["content"] for x in messages
                        if x.get("role") == "system"), None)
        msgs = [x for x in messages if x.get("role") != "system"]
        # api-top 中继的 sonnet-5-5 通道（2026-10-16 实测）：temperature
        # 非 1.0 一律 HTTP 400（上游拒绝，错误伪装成 upstream_unavailable），
        # chat/completions 与原生 /v1/messages 同病。不发该参数 = 模型默认
        # 1.0，语义等价且两个端点都收。
        payload = {"model": m, "messages": msgs, "max_tokens": max_tokens}
        if sys_msg:
            payload["system"] = sys_msg
        if r["family"]:
            payload.update(families.thinking_payload(
                families.FAMILIES[r["family"]]["thinking"], r["thinking"],
                max_tokens=max_tokens))
    else:
        payload = {"model": m, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature}
        # claude 家族走 OpenAI 兼容中继时同病：temperature 非 1.0 上游
        # 400（见上方原生分支注释），一律发模型默认
        if r["family"] == "claude":
            payload.pop("temperature", None)
        if provider == "vllm":
            payload["chat_template_kwargs"] = families.vllm_thinking(
                r["thinking"])
            # GGUF 部署的退化重复循环会以特殊 token 形式耗尽整个预算
            # （content 为空）。1.15 实测可打破循环（1.05 不够）；
            # 云端 API 不发送（OpenAI 会拒绝未知参数）
            if config.LLM_REPETITION_PENALTY > 1.0:
                payload["repetition_penalty"] = config.LLM_REPETITION_PENALTY
        elif r["family"]:
            payload.update(families.thinking_payload(
                families.FAMILIES[r["family"]]["thinking"], r["thinking"]))

    last = None
    for attempt in range(config.LLM_RETRIES):
        _rate_gate(r["url"], key=r["key"])
        try:
            req = urllib.request.Request(
                r["url"], data=json.dumps(payload).encode(), headers=headers)
            t0 = time.time()
            sock_timeout = _key_timeout(r["key"]) or config.LLM_TIMEOUT
            with urllib.request.urlopen(req, timeout=sock_timeout) as resp:
                out = json.loads(resp.read())
            # 网关过载时偶发返回畸形信封（实测 CSTCloud uni-api：message 为
            # 字符串而非对象），后续 msg.get 会以 AttributeError 打死整个
            # case——按可重试错误处理。信封形状按传输区分：OpenAI 兼容查
            # choices，Anthropic 原生查 content 块列表。
            if provider == "ollama":
                envelope_ok = (isinstance(out, dict) and isinstance(out.get("message"), dict)
                               and isinstance(out["message"].get("content"), str))
            elif provider in ("claude", "anthropic"):
                envelope_ok = (isinstance(out, dict)
                               and isinstance(out.get("content"), list))
            else:
                envelope_ok = (isinstance(out, dict)
                               and isinstance(out.get("choices"), list)
                               and out["choices"]
                               and isinstance(out["choices"][0], dict)
                               and isinstance(out["choices"][0].get("message"),
                                              dict))
            if not envelope_ok:
                raise _EnvelopeError(
                    f"malformed response envelope: {str(out)[:120]}")
            dt = time.time() - t0
            finish = ""
            if provider == "ollama":
                text = out["message"]["content"]
            elif provider in ("claude", "anthropic"):
                # Anthropic-style: content is a list of typed blocks
                text = "".join(b.get("text", "") for b in out.get("content", [])
                               if isinstance(b, dict) and b.get("type") == "text")
                finish = out.get("stop_reason", "")
            else:
                ch = out["choices"][0]
                msg = ch["message"]
                text = msg.get("content") or ""
                finish = ch.get("finish_reason") or ""
                # 部分网关（实测 CSTCloud minimax-m27）把思考以
                # <think>…</think> 内联在 content 里而非独立 reasoning 字段。
                # 必须剥离：下游 extract_json / 陷阱判定都会扫描 content，
                # 思考文本里的 JSON 片段或被篡改值会造成误评分。思考中途被
                # 截断（finish=length 且无闭合标签）时切掉尾部，空内容走
                # 下面的升格重试。
                if "<think>" in text:
                    if "</think>" in text:
                        text = re.sub(r"<think>.*?</think>", "", text,
                                      flags=re.S)
                    else:
                        text = text.split("<think>", 1)[0]
            if not str(text).strip() and finish in ("length", "max_tokens"):
                # reasoning 模型：预算被思考耗尽 → 升格重试（3×，上限 budget）
                # （OpenAI 截断符 length；Anthropic 原生为 max_tokens）
                if max_tokens >= request_ceiling:
                    raise LLMError(f"{m}: empty content, budget exhausted")
                max_tokens = min(max_tokens * 3, request_ceiling)
                if provider == "ollama":
                    payload["options"]["num_predict"] = max_tokens
                else:
                    payload["max_tokens"] = max_tokens
                last = LLMError(f"empty content (finish=length), retry @ {max_tokens}")
                continue
            rec = {"text": text, "usage": out.get("usage", {}),
                   "latency": round(dt, 1), "model": m, "provider": provider,
                   "family": r["family"], "thinking": r["thinking"]}
            _log_transcript(m, messages, rec)
            return rec
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                json.JSONDecodeError, TimeoutError, OSError,
                _EnvelopeError) as e:
            last = e
            # 网关（实测 CSTCloud uni-api）会对耗时过长的非流式生成以 422
            # "stream timeout" 中断：max_tokens 预算越大、生成越久越容易
            # 触发（32K 预算高并发下必挂，8K 秒过）。减半预算重试。
            if (isinstance(e, urllib.error.HTTPError) and e.code == 422
                    and max_tokens > 2000):
                max_tokens = max(2000, max_tokens // 2)
                if provider == "ollama":
                    payload["options"]["num_predict"] = max_tokens
                else:
                    payload["max_tokens"] = max_tokens
                last = LLMError(f"http 422, retry with max_tokens={max_tokens}")
            time.sleep(3 * (attempt + 1))
    raise LLMError(f"generation failed for {r['display']} via {provider}: {last}")


def _log_transcript(model, messages, rec):
    with _TLOCK:
        os.makedirs(os.path.dirname(TRANSCRIPT), exist_ok=True)
        ph = hashlib.md5(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()[:10]
        with open(TRANSCRIPT, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "model": model, "provider": rec["provider"],
                                "family": rec.get("family"),
                                "thinking": rec.get("thinking"),
                                "prompt_hash": ph,
                                "n_in_chars": sum(len(m.get("content", "")) for m in messages),
                                "out_chars": len(rec["text"]), **rec},
                               ensure_ascii=False) + "\n")


def _balanced_spans(text: str):
    """Yield brace-balanced {...} substrings, string/escape aware, in order."""
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break


def extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from model output.

    Try each brace-balanced object before the greedy first-{...last-} match:
    models that never emit EOS (e.g. the Baichuan-M2 GGUF deployment) repeat
    sections until the token cap, so the greedy span often concatenates two
    truncated objects and can never parse.
    """
    candidates = list(_balanced_spans(text))
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.append(m.group(0))
    for s in candidates:
        for fix in (s,
                    s.replace("，", ",").replace("：", ":"),
                    re.sub(r",\s*}", "}", s)):
            try:
                return json.loads(fix)
            except json.JSONDecodeError:
                continue
    raise LLMError(f"JSON parse failed: {text.strip()[:120]}")
