import json
import tempfile
import unittest
from pathlib import Path

from nlp.app.label_audit import build_audit_queue, load_jsonl


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

    def test_load_jsonl_reports_bad_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text('{"sample_id":"ok","query":"ok"}\n{bad}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
