import json
import tempfile
import unittest
from pathlib import Path

from nlp.app.label_audit import apply_reviewed_labels, build_audit_queue, load_jsonl
from nlp.app.run_source_label_pipeline import normalize, source_label_queue


class LabelAuditTests(unittest.TestCase):
    def _rows(self):
        return [
            {
                "sample_id": "SYN{:04d}".format(i),
                "query": "Câu hỏi {}".format(i),
                "parse_status": "PARSED",
                "parse_tree": "(S (Q_TOPIC (CHAPTER_REF CH{:02d})))".format(i % 3),
                "intent": "GET_TOPIC",
                "entities": ["CH{:02d}".format(i % 3)],
                "expected_chunk_ids": ["topics-chapter-{}".format(i % 3)],
                "expected_answerability": "EXPECTED_CONTEXT",
                "label_status": "AUTO_LABEL_REQUIRES_REVIEW",
            }
            for i in range(12)
        ]

    def test_queue_is_deterministic_and_bounded(self):
        rows = self._rows()
        first = build_audit_queue(rows, per_stratum=1, max_samples=5)
        second = build_audit_queue(rows, per_stratum=1, max_samples=5)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 5)

    def test_queue_keeps_auto_labels_separate_from_review_fields(self):
        queue = build_audit_queue(self._rows(), per_stratum=1, max_samples=10)
        self.assertTrue(queue)
        self.assertEqual(queue[0]["review"]["review_status"], "PENDING")
        self.assertIsNone(queue[0]["review"]["correct_intent"])
        self.assertEqual(queue[0]["auto_label"]["label_status"], "AUTO_LABEL_REQUIRES_REVIEW")

    def test_risky_rows_are_flagged(self):
        row = self._rows()[0]
        row["expected_answerability"] = "EXPECTED_CONTEXT"
        row["expected_chunk_ids"] = []
        queue = build_audit_queue([row])
        self.assertIn("answerable_without_expected_chunks", queue[0]["risk_flags"])

    def test_apply_reviewed_labels_exports_only_human_reviewed_rows(self):
        rows = self._rows()
        queue = build_audit_queue(rows, per_stratum=1, max_samples=3)
        queue[0]["review"].update({
            "review_status": "REVIEWED",
            "correct_intent": "GET_TOPIC",
            "correct_entities": ["CH01"],
            "correct_expected_chunk_ids": ["topics-chapter-1"],
            "correct_answerability": "EXPECTED_CONTEXT",
            "correct_parse_status": "PARSED",
            "reviewer_note": "Checked against source document",
        })
        queue[1]["review"]["review_status"] = "REJECTED"
        reviewed = apply_reviewed_labels(rows, queue)
        self.assertEqual(len(reviewed), 1)
        self.assertEqual(reviewed[0]["label_status"], "REVIEWED")
        self.assertEqual(reviewed[0]["label_provenance"], "human_review")
        self.assertEqual(reviewed[0]["entities"], ["CH01"])

    def test_apply_reviewed_labels_rejects_incomplete_review(self):
        rows = self._rows()
        queue = build_audit_queue(rows, per_stratum=1, max_samples=1)
        queue[0]["review"]["review_status"] = "REVIEWED"
        with self.assertRaisesRegex(ValueError, "missing"):
            apply_reviewed_labels(rows, queue)

    def test_load_jsonl_reports_bad_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text('{"sample_id":"ok","query":"ok"}\n{bad}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_jsonl(path)

    def test_source_label_queue_marks_all_selected_rows_reviewed(self):
        rows = self._rows()
        documents = [
            {
                "chunk_id": "topics-chapter-1",
                "kind": "topics",
                "section": "Chapter 1",
                "text": "CHAPTER 1: INTRODUCTION",
                "entity_ids": ["CH01"],
            }
        ]
        queue = build_audit_queue(rows, per_stratum=1, max_samples=3)
        source_label_queue(queue, rows, documents)
        self.assertTrue(queue)
        self.assertTrue(all(item["review"]["review_status"] == "REVIEWED" for item in queue))

    def test_multi_intent_labels_are_kept_in_audit_and_exported(self):
        rows = [{
            "sample_id": "M1", "query": "Chương 2 nói gì và LO 1.1 là gì?",
            "parse_status": "PARSED", "parse_tree": "(S_MULTI ...)",
            "intent": "MULTI_INTENT",
            "clauses": [
                {"intent": "GET_TOPIC", "entities": ["CH02"]},
                {"intent": "GET_LO", "entities": ["LO1.1"]},
            ],
            "entities": ["CH02", "LO1.1"],
            "expected_chunk_ids": ["topics-2", "lo-1.1"],
            "expected_answerability": "EXPECTED_CONTEXT",
        }]
        queue = build_audit_queue(rows)
        self.assertEqual(queue[0]["auto_label"]["clauses"], rows[0]["clauses"])
        queue[0]["review"].update({
            "review_status": "REVIEWED",
            "correct_intent": "MULTI_INTENT",
            "correct_clauses": [
                {"intent": "GET_TOPIC", "entities": ["CH02"]},
                {"intent": "GET_LO", "entities": ["LO1.1"]},
            ],
            "correct_entities": ["CH02", "LO1.1"],
            "correct_expected_chunk_ids": ["topics-2", "lo-1.1"],
            "correct_answerability": "EXPECTED_CONTEXT",
        })
        reviewed = apply_reviewed_labels(rows, queue)
        self.assertEqual(reviewed[0]["intent"], "MULTI_INTENT")
        self.assertEqual(len(reviewed[0]["clauses"]), 2)

    def test_normalize_supports_ascii_matching(self):
        self.assertEqual(normalize("Môn Cơ sở dữ liệu"), "mon co so du lieu")


if __name__ == "__main__":
    unittest.main()
