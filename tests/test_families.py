# -*- coding: utf-8 -*-
"""Family registry pure-function tests: matching, @level specs, thinking
payloads, list cap. No network involved."""
import unittest

from src.llm import families
from src.llm.client import resolve


class TestMatchFamily(unittest.TestCase):
    def test_cloud_families(self):
        cases = {"deepseek-v4-flash": "deepseek", "gpt-5": "gpt",
                 "claude-sonnet-5": "claude", "gemini-3-pro": "gemini",
                 "grok-4": "grok", "glm-5.3": "glm", "kimi-k3": "kimi",
                 "moonshot-v1-8k": "kimi", "qwen3.8-72b": "qwen",
                 "Qwen3.7-Max": "qwen"}
        for model, fam in cases.items():
            self.assertEqual(families.match_family(model), fam, model)

    def test_local_deployments_match_by_basename(self):
        # family = who the model is, independent of how it is served;
        # ollama tags (":tag") are NOT matched -- ollama is their transport
        for m, fam in (("Qwen/Qwen3.8-72B", "qwen"),
                       ("deepseek-ai/DeepSeek-V4", "deepseek"),
                       ("/mnt/models/glm5", "glm")):
            self.assertEqual(families.match_family(m), fam, m)

    def test_ollama_tags_not_matched(self):
        self.assertIsNone(families.match_family("qwen3.5:9b"))

    def test_unknown(self):
        self.assertIsNone(families.match_family("llama-4-70b"))
        self.assertIsNone(families.match_family("meta-llama/Llama-4-70B"))


class TestParseSpec(unittest.TestCase):
    def test_level_suffix(self):
        self.assertEqual(families.parse_spec("deepseek-v4-flash@high"),
                         ("deepseek-v4-flash", "high"))
        self.assertEqual(families.parse_spec("qwen3.5:9b@off"),
                         ("qwen3.5:9b", "off"))

    def test_plain(self):
        self.assertEqual(families.parse_spec("gpt-5"), ("gpt-5", None))

    def test_parse_spec_max(self):
        self.assertEqual(families.parse_spec("deepseek-v4-flash@max"),
                         ("deepseek-v4-flash", "max"))
        self.assertEqual(families.parse_spec("gpt-5@min"), ("gpt-5", "min"))

    def test_bad_level_raises(self):
        with self.assertRaises(ValueError):
            families.parse_spec("gpt-5@maxx")


class TestThinkingPayload(unittest.TestCase):
    """Adapter shapes follow each family's official docs (verified 2026-09)."""

    def test_auto_sends_nothing(self):
        for kind in ("deepseek", "qwen_cloud", "glm", "kimi", "gpt",
                     "gemini", "grok", "claude"):
            self.assertEqual(families.thinking_payload(kind, "auto"), {})

    def test_deepseek_official_mapping(self):
        # docs: low->low, medium->high, high->high, max->max
        self.assertEqual(families.thinking_payload("deepseek", "off"),
                         {"thinking": {"type": "disabled"}})
        self.assertEqual(families.thinking_payload("deepseek", "low"),
                         {"reasoning_effort": "low"})
        self.assertEqual(families.thinking_payload("deepseek", "medium"),
                         {"reasoning_effort": "high"})
        self.assertEqual(families.thinking_payload("deepseek", "max"),
                         {"reasoning_effort": "max"})

    def test_qwen_cloud_budget_only(self):
        # DashScope: enable_thinking + thinking_budget 1..32768, no tiers
        self.assertEqual(families.thinking_payload("qwen_cloud", "off"),
                         {"enable_thinking": False})
        p = families.thinking_payload("qwen_cloud", "medium")
        self.assertEqual(p, {"enable_thinking": True,
                             "thinking_budget": families.THINK_BUDGET["medium"]})
        self.assertLessEqual(
            families.thinking_payload("qwen_cloud", "max")["thinking_budget"],
            32768)

    def test_glm_flag_plus_effort(self):
        self.assertEqual(families.thinking_payload("glm", "off"),
                         {"thinking": {"type": "disabled"}})
        self.assertEqual(families.thinking_payload("glm", "max"),
                         {"thinking": {"type": "enabled"},
                          "reasoning_effort": "max"})

    def test_kimi_only_max(self):
        # K3 always thinks; reasoning_effort supports only "max"
        self.assertEqual(families.thinking_payload("kimi", "off"), {})
        self.assertEqual(families.thinking_payload("kimi", "high"), {})
        self.assertEqual(families.thinking_payload("kimi", "max"),
                         {"reasoning_effort": "max"})

    def test_gpt_minimal(self):
        self.assertEqual(families.thinking_payload("gpt", "off"),
                         {"reasoning_effort": "minimal"})
        self.assertEqual(families.thinking_payload("gpt", "max"),
                         {"reasoning_effort": "high"})

    def test_gemini_grok_no_off(self):
        self.assertEqual(families.thinking_payload("gemini", "off"), {})
        self.assertEqual(families.thinking_payload("gemini", "min"),
                         {"reasoning_effort": "minimal"})
        self.assertEqual(families.thinking_payload("grok", "max"),
                         {"reasoning_effort": "high"})

    def test_claude_budget_clamped(self):
        # native budget must stay below max_tokens
        p = families.thinking_payload("claude", "max", max_tokens=8000)
        self.assertLess(p["thinking"]["budget_tokens"], 8000)
        p2 = families.thinking_payload("claude", "low", max_tokens=8000)
        self.assertEqual(p2["thinking"]["budget_tokens"],
                         families.THINK_BUDGET["low"])
        # tiny max_tokens: no budget field, just enabled
        p3 = families.thinking_payload("claude", "high", max_tokens=1200)
        self.assertEqual(p3, {"thinking": {"type": "enabled"}})

    def test_vllm_auto_means_on(self):
        # user decision: local vllm thinking models default to thinking ON
        self.assertEqual(families.vllm_thinking("auto"),
                         {"enable_thinking": True})
        self.assertEqual(families.vllm_thinking("off"),
                         {"enable_thinking": False})
        self.assertEqual(families.vllm_thinking("high"),
                         {"enable_thinking": True,
                          "thinking_budget": families.THINK_BUDGET["high"]})


class TestResolve(unittest.TestCase):
    def test_family_endpoint_and_key_fallback(self):
        r = resolve("deepseek-v4-flash")
        self.assertEqual(r["provider"], "openai")
        self.assertEqual(r["family"], "deepseek")
        self.assertTrue(r["url"].startswith("https://api.deepseek.com"))
        # key falls back to LLM_API_KEY when family key is unset
        self.assertEqual(r["key"], families.os.getenv("LLM_API_KEY", ""))

    def test_local_transport_wins_over_cloud_endpoint(self):
        r = resolve("vllm::qwen3.8-72b", None, "http://vllm.example.com:10280/vllm/v1")
        self.assertEqual(r["provider"], "vllm")
        self.assertEqual(r["family"], "qwen")

    def test_local_deployment_keeps_family(self):
        r = resolve("Qwen/Qwen3.8-72B", "vllm",
                    "http://vllm.example.com:10280/vllm/v1")
        self.assertEqual(r["provider"], "vllm")
        self.assertEqual(r["family"], "qwen")

    def test_level_spec(self):
        r = resolve("deepseek-v4-flash@off")
        self.assertEqual(r["model"], "deepseek-v4-flash")
        self.assertEqual(r["thinking"], "off")
        self.assertEqual(r["display"], "deepseek-v4-flash@off")

    def test_local_heuristics_kept(self):
        # bare local-style names still pick their transport; they just need
        # a URL in .env (or a slot) to resolve
        with self.assertRaises(Exception):
            resolve("qwen3.5:9b", "auto")
        r = resolve("Qwen/Qwen3.6-35B", "auto",
                    "http://vllm.example.com:10280/vllm/v1")
        self.assertEqual(r["provider"], "vllm")


class TestSlots(unittest.TestCase):
    def test_parse_slots(self):
        from src.llm.families import parse_slots
        out = parse_slots([("deepseek-v4-flash", "https://api.deepseek.com/v1", "openai"),
                           ("qwen3.5:9b", None, None)])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["url_override"], "https://api.deepseek.com/v1")
        self.assertIsNone(out[1]["url_override"])

    def test_parse_slots_cap_and_empty(self):
        from src.llm.families import parse_slots
        with self.assertRaises(ValueError):
            parse_slots([(f"m{i}", None, None) for i in range(6)], limit=5)
        with self.assertRaises(ValueError):
            parse_slots([])

    def test_resolve_slot(self):
        r = resolve("deepseek-v4-flash@max", "openai",
                    "https://api.deepseek.com/v1")
        self.assertEqual(r["provider"], "openai")
        self.assertEqual(r["family"], "deepseek")
        self.assertEqual(r["thinking"], "max")
        self.assertEqual(r["url"],
                         "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(r["display"], "deepseek-v4-flash@max")

    def test_resolve_slot_ollama(self):
        r = resolve("qwen3.5:9b", "ollama", "http://127.0.0.1:11434")
        self.assertEqual(r["provider"], "ollama")
        self.assertEqual(r["url"], "http://127.0.0.1:11434/api/chat")

    def test_resolve_local_hint_without_url_raises(self):
        with self.assertRaises(Exception):  # LLMError: no URL configured
            resolve("qwen3.5:9b", "auto")


class TestParseModelList(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(families.parse_model_list("a, b@off ,c"),
                         ["a", "b@off", "c"])

    def test_cap(self):
        with self.assertRaises(ValueError):
            families.parse_model_list("a,b,c,d,e,f", limit=5)

    def test_empty(self):
        with self.assertRaises(ValueError):
            families.parse_model_list(" , ")


if __name__ == "__main__":
    unittest.main()
