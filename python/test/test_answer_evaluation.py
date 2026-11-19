"""Unit tests for the real-answer evaluation layer.

Run from python/: python -m unittest discover -s test -p "test_answer_evaluation.py" -v
"""
import unittest

from nlp.app import answer_evaluation as ev


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

    def test_unverified_reference_is_not_sent_to_judge(self):
        prompt = ev.OpenAICompatibleJudge.build_prompt(
            {
                "query": "Câu hỏi",
                "reference_answer": "Đáp án chưa kiểm chứng",
                "reference_points": ["điểm A"],
                "reference_status": "AUTO_GENERATED",
            },
            {"answer": "Câu trả lời", "status": "ANSWERED", "chunks": []},
        )
        self.assertIn('"reference_answer": null', prompt)
        self.assertIn('"reference_points": null', prompt)


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


if __name__ == "__main__":
    unittest.main()
