"""Unit tests for Member 2's modules. Run from python/:  python -m unittest tests.test_member2 -v"""
import os
import unittest

from nlp.app import answer_guard as ag
from nlp.app.prompt_builder import SENTINEL, build_prompt, citation_label

CH = [
    {"chunk_id": "schedule-week-03-3", "source": "schedule.txt", "section": "WEEK 03", "text": "WEEK 03 top-down parsing"},
    {"chunk_id": "course-info-id-1", "source": "course_info.txt", "section": "course_id: CO3085", "text": "credits: 3"},
]


class CheckAnswer(unittest.TestCase):
    def test_valid_citation_keeps_only_cited_sources(self):
        r = ag.check_answer("Tuần 3 học parsing. [Nguồn: schedule.txt / WEEK 03]", CH)
        self.assertEqual(r["status"], "ANSWERED")
        self.assertEqual(r["sources"], [{"source": "schedule.txt", "section": "WEEK 03"}])

    def test_section_containing_colon_matches(self):
        r = ag.check_answer("3 tín chỉ. [Nguồn: course_info.txt / course_id: CO3085]", CH)
        self.assertEqual(r["status"], "ANSWERED")

    def test_invented_citation_is_stripped_and_flagged(self):
        r = ag.check_answer("Hạn nộp 5/5. [Nguồn: assignments.txt / BTL1]", CH)
        self.assertEqual(r["status"], "NEEDS_REVIEW")
        self.assertNotIn("assignments.txt", r["answer"])
        self.assertEqual(r["sources"], [])

    def test_uncited_answer_needs_review(self):
        self.assertEqual(ag.check_answer("Tuần 3 học parsing.", CH)["status"], "NEEDS_REVIEW")

    def test_sentinel(self):
        r = ag.check_answer("INSUFFICIENT_CONTEXT", CH)
        self.assertEqual((r["status"], r["sources"]), ("INSUFFICIENT_CONTEXT", []))


class CheckContext(unittest.TestCase):
    found = {"retrieval_status": "FOUND", "chunks": CH}

    def test_no_chunks(self):
        r = ag.check_context("x", {"retrieval_status": "NO_RELEVANT_CONTEXT", "chunks": []}, {})
        self.assertEqual(r["status"], "NO_RELEVANT_CONTEXT")

    def test_anchored_by_intent_or_entity_is_trusted(self):
        self.assertEqual(ag.check_context("zzz qqq", self.found, {"intent": "GET_TOPIC"})["status"], "OK")
        self.assertEqual(ag.check_context("zzz qqq", self.found, {"entity_ids": ["WEEK_03"]})["status"], "OK")

    def test_unanchored_with_no_supported_word_is_refused(self):
        self.assertEqual(ag.check_context("Messi đá cho đội nào?", self.found, {})["status"], "NO_RELEVANT_CONTEXT")

    def test_unanchored_with_some_support_passes_through(self):
        self.assertEqual(ag.check_context("top-down parsing là gì?", self.found, {})["status"], "OK")

    def test_grammar_unknown_class_is_refused(self):
        r = ag.check_context("top-down parsing", self.found, {"intent": "UNKNOWN"})
        self.assertEqual(r["status"], "NO_RELEVANT_CONTEXT")


class Prompt(unittest.TestCase):
    def test_labels_sentinel_and_hint_are_not_evidence(self):
        p = build_prompt("Tuần 3 học gì?", {"intent": "GET_SCHEDULE", "entity_ids": ["WEEK_03"]}, CH)
        self.assertIn(citation_label(CH[0]), p)
        self.assertIn(SENTINEL, p)
        self.assertIn("KHÔNG phải bằng chứng", p)

    def test_no_hint_without_intent(self):
        self.assertNotIn("GỢI Ý", build_prompt("q", {}, CH))


class HonestFallback(unittest.TestCase):
    def test_missing_key_is_reported_as_fallback(self):
        from nlp.app.assistant import OpenAICompatibleLLMClient
        os.environ.pop("OPENAI_API_KEY", None)
        c = OpenAICompatibleLLMClient(model="gpt-test")
        c.generate("p", question="q", chunks=CH, semantic={})
        self.assertIn("FALLBACK", c.model_name)
        self.assertIn("gpt-test", c.model_name)


if __name__ == "__main__":
    unittest.main()
