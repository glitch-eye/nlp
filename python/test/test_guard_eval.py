"""Tests for Member 2's hallucination guard, provider wiring and measured Part IV report.
Run from python/:  python -m unittest discover -s test -v
"""
import os
import types
import unittest

from nlp.app import answer_guard as ag
from nlp.app import rag_evaluation as rag

CH = [
    {"chunk_id": "assignments-0", "source": "assignments.txt", "section": "Lưu ý",
     "text": "Dữ liệu mẫu. Bài tập lớn gồm 4 phần. Nộp qua LMS."},
    {"chunk_id": "schedule-week-05-5", "source": "schedule.txt", "section": "WEEK 05",
     "text": "WEEK 05 chapter: Chapter 4 topics 4.1 Feature systems"},
]
CITE_A = "[Nguồn: assignments.txt / Lưu ý]"
CITE_S = "[Nguồn: schedule.txt / WEEK 05]"


class UnsupportedFacts(unittest.TestCase):
    def test_fabricated_date_is_withheld_even_with_valid_citation(self):
        r = ag.check_answer("Hạn nộp là 15/12/2026. " + CITE_A, CH)
        self.assertEqual(r["status"], "NEEDS_REVIEW")
        self.assertEqual(r["unsupported_facts"], ["15/12/2026"])
        self.assertNotIn("15/12/2026", r["answer"].replace("(15/12/2026)", ""))
        self.assertIn("15/12/2026", r["raw_answer"])       # kept for the trace only
        self.assertEqual(r["sources"], [])

    def test_fabricated_time_email_url(self):
        for text in ("Nộp lúc 23:59. ", "Liên hệ a@b.vn. ", "Xem https://x.edu/y. "):
            self.assertEqual(ag.check_answer(text + CITE_A, CH)["status"], "NEEDS_REVIEW", text)

    def test_supported_numbers_pass(self):
        self.assertEqual(ag.check_answer("Bài tập lớn gồm 4 phần. " + CITE_A, CH)["status"], "ANSWERED")

    def test_leading_zero_and_decimals(self):
        r = ag.check_answer("Tuần 5 học Chương 4, mục 4.1. " + CITE_S, CH)   # WEEK 05 supports "5"
        self.assertEqual(r["status"], "ANSWERED")
        self.assertEqual(ag.check_answer("Tuần 6 học Chương 4. " + CITE_S, CH)["unsupported_facts"], ["6"])

    def test_numbers_inside_citation_labels_are_not_claims(self):
        chunks = [{"chunk_id": "x", "source": "faq.txt", "section": "Q13", "text": "Không có ngày."}]
        self.assertEqual(ag.unsupported_facts("Không có ngày. [Nguồn: faq.txt / Q13]", chunks), [])


class ProviderWiring(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.pop(k, None) for k in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY")}

    def tearDown(self):
        for k, v in self._env.items():
            if v is not None:
                os.environ[k] = v

    def test_cli_provider_and_model_are_not_ignored(self):
        from nlp.app.assistant import create_llm_client
        c = create_llm_client(provider="openai", model="gpt-test")
        self.assertTrue(c.use_api)
        self.assertEqual(c._requested_model, "gpt-test")

    def test_default_is_offline_mock(self):
        from nlp.app.assistant import create_llm_client
        self.assertIn("mock", create_llm_client().model_name)

    def test_missing_key_is_reported_not_hidden(self):
        from nlp.app.assistant import create_llm_client
        c = create_llm_client(provider="openai", model="gpt-test")
        c.generate("p", question="q", chunks=CH, semantic={})
        self.assertIn("FALLBACK", c.model_name)
        self.assertIn("gpt-test", c.model_name)
        self.assertNotIn("sk-", repr(c))


class RagEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nlp.app.assistant import CourseAssistant
        cls.assistant = CourseAssistant()
        cls.original_client = cls.assistant.llm_client
        cls.examples = rag.run_examples(cls.assistant)

    def probe(self, key):
        return next(p for p in self.examples["probes"] if key in p[0])

    def test_client_is_restored_after_probes(self):
        self.assertIs(self.assistant.llm_client, self.original_client)

    def test_probe_fabricated_date_is_blocked_end_to_end(self):
        _, _, raw, res = self.probe("bịa ngày")
        self.assertIn("15/12/2026", raw)
        self.assertEqual(res["status"], "NEEDS_REVIEW")
        self.assertEqual(res["citation_check"]["unsupported_facts"], ["15/12/2026"])

    def test_probe_invented_source_and_uncited_are_flagged(self):
        self.assertEqual(self.probe("không được gửi")[3]["status"], "NEEDS_REVIEW")
        self.assertEqual(self.probe("không trích dẫn")[3]["status"], "NEEDS_REVIEW")

    def test_probe_sentinel_gives_insufficient(self):
        self.assertEqual(self.probe("INSUFFICIENT_CONTEXT")[3]["status"], "INSUFFICIENT_CONTEXT")

    def test_deadline_question_never_reaches_the_model(self):
        res = dict(self.examples["no_info"])["mốc thời gian không có trong dữ liệu"]
        self.assertEqual(res["status"], "INSUFFICIENT_CONTEXT")

    def test_report_labels_scripted_model_and_has_no_static_scores(self):
        text = "\n".join(rag.render_sections(None, self.examples, [], "mock-x", degraded=False))
        self.assertIn("KHÔNG phải kết quả của một LLM thật", text)
        self.assertIn("III. TRƯỜNG HỢP SAI", text)

    def test_degraded_kb_is_flagged(self):
        fake = types.SimpleNamespace(kb=types.SimpleNamespace(documents=[], aliases=[]))
        self.assertTrue(rag.kb_is_degraded(fake))
        text = "\n".join(rag.render_sections(None, self.examples, [], "m", degraded=True))
        self.assertIn("CẢNH BÁO", text.splitlines()[0])

    def test_real_model_cost_cap(self):
        rows = [{"i": i} for i in range(1497)]
        self.assertEqual(len(rag._subsample(rows, rag.REAL_MODEL_SAMPLE_CAP)), 50)
        self.assertEqual(len(rag._subsample(rows, None)), 1497)
        self.assertTrue(rag.is_mock(types.SimpleNamespace(model_name="mock-grounded-extractive-v1")))
        self.assertFalse(rag.is_mock(types.SimpleNamespace(model_name="gpt-4o-mini")))

    def test_labeled_metrics_on_small_sample(self):
        rows = rag._subsample(rag.load_labeled(), 30)
        if not rows:
            self.skipTest("samples_labeled.jsonl not present")
        m = rag.labeled_metrics(rag.evaluate_labeled(self.assistant, rows))
        self.assertEqual(m["n"], 30)
        self.assertEqual(m["cited_ok"], m["answered"])     # every ANSWERED has a valid citation


if __name__ == "__main__":
    unittest.main()
