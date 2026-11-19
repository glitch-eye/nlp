import unittest

from nlp.app.evaluation import evaluate_parser, evaluate_semantics
from nlp.app.parser import ParseNode


def single_tree(intent_label):
    return ParseNode("S", (ParseNode(intent_label),))


def multi_tree():
    return ParseNode("S_MULTI", (
        ParseNode("CLAUSE", (single_tree("Q_TOPIC"),)),
        ParseNode("CLAUSE", (single_tree("Q_LO"),)),
    ))


class MultiIntentEvaluationTests(unittest.TestCase):
    class Parser:
        def parse(self, query):
            raise AssertionError("Evaluation should prefer parse_multi when available")

        def parse_multi(self, query):
            return multi_tree() if "và" in query else single_tree("Q_TOPIC")

    def test_parser_evaluation_uses_multi_tree_without_changing_single_tree(self):
        rows = [
            {"sample_id": "S1", "query": "Chương 2 nói gì?", "parse_status": "PARSED",
             "parse_tree": single_tree("Q_TOPIC").to_bracketed()},
            {"sample_id": "M1", "query": "Chương 2 nói gì và LO 1.1 là gì?",
             "parse_status": "PARSED", "parse_tree": multi_tree().to_bracketed()},
        ]
        result = evaluate_parser(rows, self.Parser())
        self.assertEqual(result["evaluated"], 2)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["tree_exact_match_rate"], 1.0)

    def test_semantic_evaluation_compares_multi_clause_intent_set(self):
        rows = [{
            "sample_id": "M1",
            "query": "Chương 2 nói gì và LO 1.1 là gì?",
            "intent": "MULTI_INTENT",
            "clauses": [{"intent": "GET_TOPIC"}, {"intent": "GET_LO"}],
            "entities": [],
        }]
        result = evaluate_semantics(rows, self.Parser())
        self.assertEqual(result["intent_evaluated"], 1)
        self.assertEqual(result["intent_correct"], 1)
        self.assertEqual(result["intent_accuracy"], 1.0)
        self.assertEqual(result["failures"], [])

    def test_legacy_single_intent_schema_still_matches(self):
        rows = [{
            "sample_id": "S1", "query": "Chương 2 nói gì?",
            "intent": "GET_TOPIC", "entities": [],
        }]
        result = evaluate_semantics(rows, self.Parser())
        self.assertEqual(result["intent_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
