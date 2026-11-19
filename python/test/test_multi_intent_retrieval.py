import unittest

from nlp.app.retriever import Retriever


class MultiIntentRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"chunk_id": "shared", "text": "Course schedule and topic information for chapter 4 week 4.", "source": "course.txt", "section": "shared", "kind": "faq", "entity_ids": ["CH04", "WEEK_04"]},
            {"chunk_id": "topic", "text": "Chapter 4 covers parsing and grammar topics.", "source": "topics.txt", "section": "CHAPTER 4", "kind": "topics", "entity_ids": ["CH04"]},
            {"chunk_id": "schedule", "text": "Week 4 schedule covers chapter 4 parsing.", "source": "schedule.txt", "section": "WEEK 04", "kind": "schedule", "entity_ids": ["CH04", "WEEK_04"]},
            {"chunk_id": "unrelated", "text": "Assignment submission requirements and output files.", "source": "assignments.txt", "section": "PART I", "kind": "assignments", "entity_ids": ["PART_I"]},
        ]
        self.retriever = Retriever(self.documents, parser=object(), top_k=2, min_score=0.0, min_lexical_score=0.0)

    def test_multi_retrieval_returns_one_deduplicated_global_evidence_pack(self):
        clauses = ["What topics are in chapter 4?", "What is the schedule for week 4?"]
        frames = [
            {"intent": "GET_TOPIC", "entity_ids": ["CH04"], "parse_status": "PARSED"},
            {"intent": "GET_SCHEDULE", "entity_ids": ["WEEK_04"], "parse_status": "PARSED"},
        ]
        result = self.retriever.retrieve_multi("combined question", clauses, frames)
        ids = [chunk["chunk_id"] for chunk in result["chunks"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), 2)
        self.assertEqual(result["intent"], "MULTI_INTENT")
        self.assertEqual(result["intent_coverage"], 1.0)
        self.assertEqual(set(result["covered_clause_indices"]), {0, 1})
        self.assertEqual(result["selected_chunk_count"], len(ids))
        self.assertTrue(result["chunk_support"])

    def test_multi_retrieval_validates_clause_frame_count(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve_multi("q", ["one", "two"], [{"intent": "GET_TOPIC", "entity_ids": []}])

    def test_empty_multi_query_has_no_evidence(self):
        result = self.retriever.retrieve_multi("q", [], [])
        self.assertEqual(result["retrieval_status"], "NO_RELEVANT_CONTEXT")
        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["intent_coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
