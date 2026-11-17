"""Regression tests for safe multi-intent parsing and semantic composition."""
import unittest

from nlp.app.parser import CourseParser
from nlp.app.semantic_frame import frame_from_tree
from nlp.app.multi_intent import split_intent_clauses


class MultiIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = CourseParser()

    def test_two_independent_intents_are_split(self):
        query = "Chương 4 nói về gì và Tuần 5 học gì?"
        clauses = split_intent_clauses(query, self.parser)
        self.assertEqual(len(clauses), 2)
        self.assertIsNotNone(self.parser.parse(clauses[0]))
        self.assertIsNotNone(self.parser.parse(clauses[1]))

    def test_parse_clauses_api_returns_trees(self):
        result = self.parser.parse_clauses(
            "Chương 4 nói về gì và Tuần 5 học gì?"
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(all(tree is not None for _, tree in result))

    def test_multi_parse_tree_and_semantic_frames(self):
        query = "Chương 4 nói về gì và Tuần 5 học gì?"
        tree = self.parser.parse_multi(query)
        self.assertEqual(tree.label, "S_MULTI")
        frame = frame_from_tree(tree, query, ())
        self.assertEqual(frame["intent"], "MULTI_INTENT")
        self.assertEqual(
            [clause["intent"] for clause in frame["clauses"]],
            ["GET_TOPIC", "GET_SCHEDULE"],
        )
        self.assertIn("CH04", frame["clauses"][0]["entity_ids"])
        self.assertIn("WEEK_05", frame["clauses"][1]["entity_ids"])

    def test_single_intent_query_is_unchanged(self):
        query = "Chương 3 và Chương 4 khác nhau như thế nào?"
        self.assertEqual(split_intent_clauses(query, self.parser), [query])
        frame = frame_from_tree(self.parser.parse(query), query, ())
        self.assertIn("CH03", frame["entity_ids"])
        self.assertIn("CH04", frame["entity_ids"])

    def test_lexical_topic_conjunction_is_not_split(self):
        query = "Tài liệu nào nói về CFG và Parsing?"
        self.assertEqual(split_intent_clauses(query, self.parser), [query])

    def test_unrecognized_clause_does_not_partially_split(self):
        query = "Chương 4 nói về gì và thời tiết hôm nay thế nào?"
        self.assertEqual(split_intent_clauses(query, self.parser), [query])


if __name__ == "__main__":
    unittest.main()
