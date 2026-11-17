"""Tests for semantic.py and llm_client.py (Member 2). Run from python/: python -m unittest discover -s test -v"""
import os
import unittest
from unittest import mock

from nlp.app.parser import CourseParser
from nlp.app import semantic
from nlp.app.llm_client import LLMClient
from nlp.app.prompt_builder import SENTINEL


class Interpret(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = CourseParser()

    def frame(self, q):
        return semantic.interpret(self.parser.parse(q), q, aliases=())

    def test_unparsed_has_no_intent(self):
        f = semantic.interpret(None, "blah")
        self.assertEqual((f["parse_status"], f["intent"], f["entity_ids"]), ("UNPARSED", None, []))

    def test_chapter_question(self):
        f = self.frame("Chương 4 nói về gì?")
        self.assertEqual(f["parse_status"], "PARSED")
        self.assertEqual(f["intent"], "GET_TOPIC")
        self.assertEqual(f["entities"].get("chapter"), "CH04")

    def test_week_question(self):
        f = self.frame("Tuần 3 học gì?")
        self.assertIn("WEEK_03", f["entity_ids"])
        self.assertEqual(f["entities"].get("week"), "WEEK_03")

    def test_category_helper(self):
        self.assertEqual(semantic.entity_category("CH04"), "chapter")
        self.assertEqual(semantic.entity_category("LO1.2"), "lo")
        self.assertEqual(semantic.entity_category(None), None)


class Client(unittest.TestCase):
    def env(self, **kw):
        base = {k: v for k, v in os.environ.items()
                if k not in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL")}
        base.update(kw)
        return mock.patch.dict(os.environ, base, clear=True)

    def test_default_is_offline_mock(self):
        with self.env():
            c = LLMClient()
            out = c.generate("p", question="q", chunks=[], semantic={})
        self.assertFalse(c.use_api)
        self.assertIn("mock", c.model_name)
        self.assertIsInstance(out, str)

    def test_real_provider_without_key_is_flagged_fallback(self):
        with self.env(LLM_PROVIDER="openai", LLM_MODEL="gpt-test"):
            c = LLMClient()
            c.generate("p", question="q", chunks=[], semantic={})
        self.assertIn("FALLBACK", c.model_name)
        self.assertIn("gpt-test", c.model_name)
        self.assertEqual(c.last_error, "OPENAI_API_KEY not set")

    def test_auto_without_key_stays_mock(self):
        with self.env(LLM_PROVIDER="auto"):
            self.assertFalse(LLMClient().use_api)

    def test_network_error_falls_back_and_hides_key(self):
        import urllib.error
        with self.env(LLM_PROVIDER="openai", OPENAI_API_KEY="sk-secret"):
            c = LLMClient()
            with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
                c.generate("p", question="q", chunks=[], semantic={})
            self.assertIn("FALLBACK", c.model_name)
            self.assertNotIn("sk-secret", c.model_name + repr(c) + str(c.last_error))

    def test_success_reports_requested_model(self):
        import io, json
        body = json.dumps({"choices": [{"message": {"content": "  ok [Nguồn: a / b] "}}]}).encode()
        fake = mock.MagicMock()
        fake.__enter__.return_value = io.BytesIO(body)
        with self.env(LLM_PROVIDER="openai", OPENAI_API_KEY="k", LLM_MODEL="m1"):
            c = LLMClient()
            with mock.patch("urllib.request.urlopen", return_value=fake):
                out = c.generate("p")
        self.assertEqual(out, "ok [Nguồn: a / b]")
        self.assertEqual(c.model_name, "m1")


if __name__ == "__main__":
    unittest.main()
