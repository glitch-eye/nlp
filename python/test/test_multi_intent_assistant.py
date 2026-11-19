import unittest

from nlp.app.assistant import CourseAssistant
from nlp.app.parser import ParseNode


class FakeParser:
    def parse(self, text):
        label = "Q_TOPIC" if "topic" in text.casefold() else "Q_SCHEDULE"
        return ParseNode("S", (ParseNode(label),))


class FakeRetriever:
    def __init__(self):
        self.calls = 0
    def retrieve_multi(self, question, clauses, clause_frames):
        self.calls += 1
        chunks = [{"chunk_id": "c1", "text": "Topic and schedule evidence.", "source": "kb.txt", "section": "all", "kind": "faq", "entity_ids": []}]
        return {
            "chunks": chunks, "scores": [0.9], "retrieval_status": "FOUND",
            "clauses": [
                {"retrieval_status": "FOUND", "candidate_chunk_ids": ["c1"]},
                {"retrieval_status": "FOUND", "candidate_chunk_ids": ["c1"]},
            ],
            "intent_coverage": 1.0, "chunk_support": [{"chunk_id": "c1", "clause_indices": [0, 1]}],
            "candidate_chunk_count": 1,
        }


class FakeLLM:
    model_name = "fake"
    def __init__(self):
        self.generation_count = 0
    def generate(self, prompt, question="", chunks=None, semantic=None):
        self.generation_count += 1
        return "Combined answer"


class MultiIntentAssistantTests(unittest.TestCase):
    def test_multi_intent_uses_one_global_retrieval_and_one_generation_call(self):
        assistant = CourseAssistant.__new__(CourseAssistant)
        assistant.parser = FakeParser()
        assistant.kb = type("KB", (), {"aliases": []})()
        assistant.retriever = FakeRetriever()
        assistant.llm_client = FakeLLM()
        assistant._teammate_semantic = None
        assistant._teammate_prompt_builder = None
        assistant._teammate_answer_guard = None

        result = assistant.answer("What topic? và What schedule?", dialogue_state={})

        self.assertEqual(assistant.retriever.calls, 1)
        self.assertEqual(assistant.llm_client.generation_count, 1)
        self.assertEqual(result["model_call_count"], 1)
        self.assertEqual(result["intent"], "MULTI_INTENT")
        self.assertEqual(result["intent_coverage"], 1.0)
        self.assertEqual(len(result["chunks"]), 1)


if __name__ == "__main__":
    unittest.main()
