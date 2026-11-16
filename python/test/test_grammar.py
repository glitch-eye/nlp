"""Grammar tests. From python/ run:  python -m unittest discover -s test -v

1. No regression: every sentence from the previous sample set still parses to the SAME tree.
2. Coverage: the dialogue questions that used to be UNPARSED now parse, with the right intent/entities.
3. Soundness: everything the grammar generates parses back; out-of-domain behaviour is unchanged.
"""
import json
import unittest
from pathlib import Path

from nlp.app.course_grammar import CourseGrammar
from nlp.app.parser import CourseParser
from nlp.app.semantic_frame import frame_from_tree
from nlp.app.sentence_generator import SentenceGenerator

APP = Path(__file__).resolve().parents[1] / "nlp" / "app"
PARSER = CourseParser()


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)


def frame(question):
    tree = PARSER.parse(question)
    return None if tree is None else frame_from_tree(tree, question, ())


class NoRegression(unittest.TestCase):
    def test_previous_samples_keep_identical_trees(self):
        path = APP / "output" / "samples_labeled.jsonl"
        if not path.exists():
            self.skipTest("samples_labeled.jsonl not present")
        bad = []
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["parse_status"] != "PARSED":
                continue
            if PARSER.parse_line(rec["query"]) != rec["parse_tree"]:
                bad.append(rec["query"])
        self.assertEqual(bad[:5], [], "{} sentences changed tree".format(len(bad)))

    def test_project_sentences_still_parse(self):
        for q in (APP / "input" / "sentences.txt").read_text(encoding="utf-8").splitlines():
            self.assertIsNotNone(PARSER.parse(q), q)


class NewCoverage(unittest.TestCase):
    CASES = [
        # question, intent, entities that must be present
        ("Chương 4 có liên quan đến parsing không?", "GET_TOPIC", {"CH04"}),
        ("Chương 11 có hỏi đáp không?", "GET_TOPIC", {"CH11"}),
        ("Chương 3 và Chương 4 khác nhau như thế nào?", "GET_TOPIC", {"CH03"}),
        ("Bài tập lớn có phần parser không?", "GET_ASSIGNMENT", set()),
        ("Bài tập lớn có phần hội thoại không?", "GET_ASSIGNMENT", set()),
        ("Tài liệu tham khảo cho Chương 5 là gì?", "GET_RESOURCE", {"CH05"}),
        ("Tài liệu tham khảo cho Tuần 5 là gì?", "GET_RESOURCE", {"WEEK_05"}),
        ("tài liệu cho Phần II ở đâu?", "GET_RESOURCE", set()),
    ]

    def test_parse_intent_entities(self):
        for q, intent, ents in self.CASES:
            f = frame(q)
            self.assertIsNotNone(f, "UNPARSED: " + q)
            self.assertEqual(f["intent"], intent, q)
            self.assertTrue(ents <= set(f["entity_ids"]), (q, f["entity_ids"]))

    def test_topic_is_a_TOPIC_REF_node(self):
        # the semantic frame only extracts topic entities from TOPIC_REF nodes
        tree = PARSER.parse("Chương 4 có liên quan đến parsing không?")
        topic = [n for n in _walk(tree) if n.label == "TOPIC_REF"]
        self.assertEqual(len(topic), 1)
        self.assertEqual(topic[0].children[0].label.casefold(), "parsing")

    def test_still_rejected(self):
        for q in ("Chương 4 có liên quan không?", "Tài liệu cho Chương mười là gì?",
                  "Chương 13 có hỏi đáp không?", "Bài tập lớn có phần không?", "Thời tiết hôm nay thế nào?"):
            self.assertIsNone(PARSER.parse(q), q)


class Soundness(unittest.TestCase):
    def test_every_generated_sentence_parses_to_a_known_intent(self):
        grammar = CourseGrammar("vi")
        sents = SentenceGenerator(grammar).generate(max_sentences=10000)
        self.assertLess(len(sents), 10000, "generator hit the cap; samples would be truncated")
        unknown = []
        for s in sents:
            tree = PARSER.parse(s)
            self.assertIsNotNone(tree, s)
            if frame_from_tree(tree, s, ())["intent"] == "UNKNOWN":
                unknown.append(s)
        self.assertEqual(len(unknown), 3, unknown[:5])  # only the 3 canned out-of-domain sentences

    def test_out_of_domain_samples_unchanged(self):
        for q in ("Hôm nay ở TP.HCM có mưa không?", "Ai là tổng thống của Pháp?", "Hãy giải bài tập xác suất này giúp tôi."):
            self.assertEqual(frame(q)["intent"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
