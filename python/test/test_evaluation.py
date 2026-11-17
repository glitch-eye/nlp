"""Tests for the mainline evaluation module.

Run from python/: python -m unittest discover -s test -p test_evaluation.py -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from nlp.app.evaluation import (
    _ratio,
    build_report,
    evaluate_parser,
    evaluate_semantics,
    load_samples,
    render_markdown,
    subsample,
)
from nlp.app.parser import CourseParser


class EvaluationUtilities(unittest.TestCase):
    def test_ratio_handles_empty_denominator(self):
        self.assertIsNone(_ratio(0, 0))
        self.assertEqual(_ratio(3, 4), 0.75)

    def test_subsample_is_stable_and_respects_limit(self):
        rows = list(range(100))
        selected = subsample(rows, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(selected, subsample(rows, 10))
        self.assertEqual(subsample(rows, None), rows)

    def test_load_samples_reports_invalid_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"query":"ok"}\n{bad json}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":2:"):
                load_samples(path)

    def test_load_samples_validates_required_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps({"query": "q", "entities": "CH04"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be lists"):
                load_samples(path)


class MainlineStageMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = CourseParser()

    def test_parser_reports_parse_status_and_tree_regression(self):
        query = "Chương 4 nói về gì?"
        tree = self.parser.parse(query)
        self.assertIsNotNone(tree)
        samples = [{
            "sample_id": "P1", "query": query,
            "parse_status": "PARSED", "parse_tree": tree.to_bracketed(),
        }]
        metrics = evaluate_parser(samples, self.parser)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["tree_exact_match_rate"], 1.0)
        self.assertEqual(metrics["failures"], [])

    def test_parser_records_mismatch(self):
        metrics = evaluate_parser([{
            "sample_id": "P2", "query": "Thời tiết hôm nay thế nào?",
            "parse_status": "PARSED",
        }], self.parser)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["failures"][0]["actual_parse_status"], "UNPARSED")

    def test_semantic_intent_and_entity_metrics(self):
        query = "Chương 4 nói về gì?"
        metrics = evaluate_semantics([{
            "sample_id": "S1", "query": query,
            "intent": "GET_TOPIC", "entities": ["CH04"],
        }], self.parser)
        self.assertEqual(metrics["intent_accuracy"], 1.0)
        self.assertEqual(metrics["entity_precision_micro"], 1.0)
        self.assertEqual(metrics["entity_recall_micro"], 1.0)
        self.assertEqual(metrics["entity_f1_micro"], 1.0)


class ReportTests(unittest.TestCase):
    def test_report_contains_provenance_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text('{"query":"q"}\n', encoding="utf-8")
            samples = [{
                "sample_id": "1", "query": "q", "intent": "UNKNOWN",
                "entities": [], "expected_chunk_ids": [],
                "expected_answerability": "NO_RELEVANT_CONTEXT",
                "label_status": "MANUAL_REVIEWED",
            }]
            parser_metrics = {
                "accuracy": 1.0, "tree_exact_match_rate": 1.0,
                "tree_exact_match_count": 1, "tree_exact_match_evaluated": 1,
            }
            semantic_metrics = {
                "intent_accuracy": 1.0, "entity_precision_micro": 1.0,
                "entity_recall_micro": 1.0, "entity_f1_micro": 1.0,
            }
            report = build_report(
                samples, path, parser_metrics, semantic_metrics,
                rag_evidence=["RAG evidence probe: scripted model rejected by guard"],
            )
            markdown = render_markdown(report)
            self.assertEqual(report["dataset"]["sample_count"], 1)
            self.assertIn("RAG evidence probe: scripted model rejected by guard", report["rag_evidence"])
            self.assertEqual(len(report["dataset"]["sha256"]), 64)
            self.assertIn("## Parser", markdown)
            self.assertIn("## Semantic interpretation", markdown)
            self.assertIn("AUTO_LABEL_REQUIRES_REVIEW", markdown)


if __name__ == "__main__":
    unittest.main()
