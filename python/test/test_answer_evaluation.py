"""Unit tests for the real-answer evaluation layer.

Run from python/: python -m unittest discover -s test -p "test_answer_evaluation.py" -v
"""
import unittest
from unittest.mock import patch

from nlp.app import answer_evaluation as ev
from nlp.app.llm_client import LLMClient


class DatasetTests(unittest.TestCase):
    def test_stratified_sampling_keeps_answerability_groups(self):
        rows = [
            {"sample_id": "a1", "expected_answerability": "EXPECTED_CONTEXT"},
            {"sample_id": "a2", "expected_answerability": "EXPECTED_CONTEXT"},
            {"sample_id": "b1", "expected_answerability": "INSUFFICIENT_CONTEXT"},
            {"sample_id": "c1", "expected_answerability": "REVIEW"},
        ]
        selected = ev.stratified_sample(rows, 3)
        self.assertEqual({row["sample_id"] for row in selected}, {"a1", "b1", "c1"})

    def test_judge_json_validation(self):
        result = ev.parse_judge_json(
            '{"scores":{"correctness":4,"faithfulness":5,"relevancy":4,'
            '"completeness":3},"hallucination":false,"reason":"supported"}'
        )
        self.assertEqual(result["scores"]["faithfulness"], 5.0)
        self.assertFalse(result["hallucination"])

    def test_judge_rejects_out_of_range_score(self):
        with self.assertRaises(ValueError):
            ev.parse_judge_json(
                '{"scores":{"correctness":6,"faithfulness":5,"relevancy":4,'
                '"completeness":3},"hallucination":false}'
            )

    def test_judge_sees_raw_answer_blocked_by_guard(self):
        prompt = ev.OpenAICompatibleJudge.build_prompt(
            {
                "query": "Khi nào deadline?",
                "expected_answerability": "EXPECTED_CONTEXT",
            },
            {
                "answer": "Câu trả lời bị giữ lại vì thiếu bằng chứng.",
                "status": "NEEDS_REVIEW",
                "citation_check": {"raw_answer": "Deadline là 15/12/2026."},
                "chunks": [],
            },
        )
        self.assertIn("Deadline là 15/12/2026.", prompt)
        self.assertIn("Câu trả lời bị giữ lại vì thiếu bằng chứng.", prompt)


    def test_unverified_reference_is_not_sent_to_judge(self):
        prompt = ev.OpenAICompatibleJudge.build_prompt(
            {
                "query": "Câu hỏi",
                "reference_answer": "Đáp án chưa kiểm chứng",
                "reference_points": ["điểm A"],
                "reference_status": "AUTO_GENERATED",
                "expected_answerability": "EXPECTED_CONTEXT",
                "label_status": "GOLD_REVIEWED",
                "label_provenance": "auto_rule",
            },
            {"answer": "Câu trả lời", "status": "ANSWERED", "chunks": []},
        )
        self.assertIn('"reference_answer": null', prompt)
        self.assertIn('"reference_points": null', prompt)
        self.assertIn('"expected_answerability": null', prompt)


class JudgeIsolationTests(unittest.TestCase):
    def test_mock_answer_is_not_sent_to_judge(self):
        class MockClient:
            model_name = "mock-grounded-extractive-v1"
            last_error = None

        class Assistant:
            llm_client = MockClient()

            def answer(self, query):
                return {
                    "answer": "mock answer",
                    "status": "ANSWERED",
                    "retrieved_chunk_ids": [],
                    "chunks": [],
                }

        class Judge:
            called = False

            def evaluate(self, row, output):
                self.called = True
                raise AssertionError("must not judge a mock answer")

        judge = Judge()
        result = ev.evaluate_rows(
            Assistant(),
            [{"sample_id": "x", "query": "question", "expected_answerability": "REVIEW"}],
            judge=judge,
        )[0]
        self.assertFalse(judge.called)
        self.assertIsNone(result["judge"])
        self.assertTrue(result["judge_error"].startswith("SKIPPED:"))


class MetricsTests(unittest.TestCase):
    def test_only_human_reviewed_labels_enter_headline_metrics(self):
        results = [
            {
                "sample_id": "gold",
                "expected_answerability": "EXPECTED_CONTEXT",
                "label_status": "REVIEWED",
                "label_provenance": "human_review",
                "expected_chunk_ids": ["c1"],
                "retrieved_chunk_ids": ["c1"],
                "actual_status": "ANSWERED",
                "citation_check": {"valid_citations": ["x"], "invalid_citations": []},
                "judge": None,
                "model_fallback": False,
            },
            {
                "sample_id": "auto",
                "expected_answerability": "EXPECTED_CONTEXT",
                "label_status": "GOLD_REVIEWED",
                "label_provenance": "auto_rule",
                "expected_chunk_ids": ["c2"],
                "retrieved_chunk_ids": [],
                "actual_status": "NEEDS_REVIEW",
                "citation_check": {},
                "judge": None,
                "model_fallback": False,
            },
        ]
        metrics = ev.calculate_metrics(results)
        self.assertEqual(metrics["human_reviewed_label_count"], 1)
        self.assertEqual(metrics["answerable_count"], 1)
        self.assertEqual(metrics["retrieval_hit_at_k"], 1.0)
        self.assertEqual(metrics["unverified_label_count"], 1)

    def test_fallback_count_is_reported(self):
        results = [{
            "expected_answerability": "REVIEW",
            "label_status": "AUTO",
            "label_provenance": "auto",
            "expected_chunk_ids": [],
            "retrieved_chunk_ids": [],
            "actual_status": "ANSWERED",
            "citation_check": {},
            "judge": None,
            "model_fallback": True,
        }]
        self.assertEqual(ev.calculate_metrics(results)["fallback_count"], 1)



class LLMClientTelemetryTests(unittest.TestCase):
    def test_mock_call_is_recorded_without_exposing_prompt(self):
        client = LLMClient(provider="mock")
        client.generate("private prompt", question="private question")
        self.assertEqual(client.generation_count, 1)
        self.assertEqual(len(client.call_history), 1)
        call = client.last_call
        self.assertEqual(call["provider"], "mock")
        self.assertFalse(call["used_api"])
        self.assertFalse(call["fallback"])
        self.assertNotIn("prompt", call)
        self.assertNotIn("private", repr(call))

    def test_successful_api_call_is_marked_real(self):
        client = LLMClient(provider="openai", model="test-model", api_key="not-a-real-key")
        with patch.object(client, "_call_api", return_value="generated"):
            self.assertEqual(client.generate("prompt"), "generated")
        self.assertTrue(client.last_call["used_api"])
        self.assertFalse(client.last_call["fallback"])
        self.assertEqual(client.last_call["model_name"], "test-model")

    def test_missing_api_key_is_recorded_as_fallback(self):
        client = LLMClient(provider="openai", model="test-model", api_key="")
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            client._api_key = ""
            client.generate("prompt")
        self.assertTrue(client.last_call["fallback"])
        self.assertFalse(client.last_call["used_api"])
        self.assertEqual(client.last_call["error"], "OPENAI_API_KEY not set")


class MultiIntentProvenanceTests(unittest.TestCase):
    def test_mixed_model_calls_are_reported_and_not_sent_to_judge(self):
        class Client:
            model_name = "mock-fallback"
            last_error = "TimeoutError"
            call_history = []

        class Assistant:
            llm_client = Client()

            def answer(self, query):
                self.llm_client.call_history.extend([
                    {"provider": "openai", "model_name": "gpt-test", "used_api": True,
                     "fallback": False, "error": None, "elapsed_seconds": 0.4},
                    {"provider": "openai", "model_name": "mock [FALLBACK: TimeoutError]",
                     "used_api": False, "fallback": True, "error": "TimeoutError",
                     "elapsed_seconds": 0.01},
                ])
                return {"answer": "combined", "status": "ANSWERED", "chunks": []}

        class Judge:
            called = False
            def evaluate(self, row, output):
                self.called = True
                raise AssertionError("mixed API/fallback output must not be judged as real-only")

        judge = Judge()
        result = ev.evaluate_rows(
            Assistant(), [{"sample_id": "multi", "query": "q", "expected_answerability": "REVIEW"}],
            judge=judge,
        )[0]
        self.assertEqual(result["model_execution"], "mixed")
        self.assertEqual(result["model_call_count"], 2)
        self.assertTrue(result["model_fallback"])
        self.assertAlmostEqual(result["generation_elapsed_seconds"], 0.41)
        self.assertFalse(judge.called)
        self.assertTrue(result["judge_error"].startswith("SKIPPED:"))


if __name__ == "__main__":
    unittest.main()
