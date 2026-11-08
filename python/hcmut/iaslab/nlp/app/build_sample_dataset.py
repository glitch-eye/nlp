"""Generate bounded grammar examples, parse them, and attach semantic labels."""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from iaslab.nlp.app.knowledge_base import KnowledgeBase, default_data_dir, load_entity_aliases
from iaslab.nlp.app.parser import CourseParser, ParseNode
from iaslab.nlp.app.sentence_generator import SentenceGenerator


INTENT_TO_KIND = {
    "GET_COURSE_INFO": "course_info",
    "GET_SCHEDULE": "schedule",
    "GET_TOPIC": "topics",
    "GET_ASSIGNMENT": "assignments",
    "GET_DEADLINE": "assignments",
    "GET_RESOURCE": "resources",
    "GET_RULE": "regulations",
    "GET_LO": "course_info",
}
TREE_INTENT_TO_INTENT = {
    "Q_COURSE_INFO": "GET_COURSE_INFO",
    "Q_SCHEDULE": "GET_SCHEDULE",
    "Q_TOPIC": "GET_TOPIC",
    "Q_ASSIGNMENT": "GET_ASSIGNMENT",
    "Q_DEADLINE": "GET_DEADLINE",
    "Q_RESOURCE": "GET_RESOURCE",
    "Q_RULE": "GET_RULE",
    "Q_LO": "GET_LO",
}


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def walk(node):
    yield node
    for child in node.children:
        if isinstance(child, ParseNode):
            for descendant in walk(child):
                yield descendant


def node_text(node):
    return " ".join(child.label for child in walk(node) if not child.children).strip()


def first_node(root, label):
    return next((node for node in walk(root) if node.label == label), None)


def _add_alias_entity(text, aliases, found):
    normalized_text = " " + normalize(text) + " "
    for alias, entity_id, canonical in aliases:
        for phrase in (alias, canonical):
            normalized_phrase = " " + normalize(phrase) + " "
            if normalized_phrase.strip() and normalized_phrase in normalized_text:
                if entity_id not in found:
                    found.append(entity_id)


def semantic_labels(tree, aliases):
    intent = next((TREE_INTENT_TO_INTENT.get(node.label, "UNKNOWN") for node in tree.children
                   if isinstance(node, ParseNode) and node.label.startswith("Q_")), "UNKNOWN")
    entities = []

    for label in ("COURSE_REF", "EXAM_REF", "WEEK_REF", "CHAPTER_REF", "TOPIC_REF",
                  "ASSIGNMENT_REF", "PART_REF", "RESOURCE_TOPIC", "RESOURCE_REF",
                  "RULE_REF", "RULE_OBJECT", "LO_REF", "DEADLINE_REF"):
        node = first_node(tree, label)
        if node is None:
            continue
        value = node_text(node)
        if label == "COURSE_REF":
            entities.append("CO3085")
        elif label == "EXAM_REF":
            entities.append("MIDTERM" if "midterm" in value.casefold() or "giữa kỳ" in value.casefold() else "FINAL")
        elif label == "WEEK_REF":
            number = re.search(r"\d+", value)
            if number:
                entities.append("WEEK_{:02d}".format(int(number.group(0))))
        elif label == "CHAPTER_REF":
            number = re.search(r"\d+", value)
            if number:
                entities.append("CH{:02d}".format(int(number.group(0))))
        elif label == "LO_REF":
            match = re.search(r"\d+\.\d+", value)
            if match:
                entities.append("LO" + match.group(0))
        elif label in ("ASSIGNMENT_REF", "DEADLINE_REF"):
            entities.append("BTL01")
        elif label == "RULE_OBJECT":
            entities.append("LLM_RAG" if "llm" in value.casefold() else "AI_USAGE")
        elif label == "RULE_REF":
            entities.append("LLM_RAG" if "llm/rag" in value.casefold() else "AI_USAGE")
        elif label == "RESOURCE_TOPIC":
            normalized = normalize(value)
            entities.append(
                "PYTHON_NLP" if "python" in normalized else
                "STATISTICAL_NLP" if "statistical" in normalized or "thong ke" in normalized else
                "MACHINE_TRANSLATION"
            )
        else:
            _add_alias_entity(value, aliases, entities)

    return intent, list(dict.fromkeys(entities))


def expected_chunks(intent, entity_ids, documents):
    if intent == "GET_DEADLINE":
        # The sample KB says deadlines are sample data but gives no due date.
        return []
    kind = INTENT_TO_KIND.get(intent)
    if not kind:
        return []
    candidates = [doc for doc in documents if doc["kind"] == kind]
    matched = [doc["chunk_id"] for doc in candidates
               if set(entity_ids).intersection(doc.get("entity_ids", []))]
    resource_terms = {
        "STATISTICAL_NLP": ("statistical", "statistics"),
        "PYTHON_NLP": ("python",),
        "MACHINE_TRANSLATION": ("machine translation",),
    }
    if intent == "GET_RESOURCE" and not matched:
        terms = tuple(term for entity in entity_ids for term in resource_terms.get(entity, ()))
        matched = [doc["chunk_id"] for doc in candidates
                   if any(term in normalize(doc["text"]) for term in terms)]
    if not matched and intent in ("GET_COURSE_INFO", "GET_SCHEDULE", "GET_ASSIGNMENT",
                                  "GET_RESOURCE", "GET_RULE", "GET_LO"):
        matched = [doc["chunk_id"] for doc in candidates]
    return matched


def build_records(grammar_path, data_dir, max_sentences=10000):
    course_parser = CourseParser(grammar_path)
    raw_rules = course_parser.rules

    class FileGrammar(object):
        start_symbol = "S"

        def is_nonterminal(self, symbol):
            return symbol in raw_rules

        def productions_for(self, nonterminal):
            return [[symbol for _, symbol in production]
                    for production in raw_rules.get(nonterminal, ())]

    sentences = SentenceGenerator(FileGrammar()).generate(max_sentences=max_sentences)
    knowledge_base = KnowledgeBase(data_dir)
    aliases = load_entity_aliases(Path(data_dir) / "entities.txt")
    records = []
    for index, sentence in enumerate(sentences, 1):
        tree = course_parser.parse(sentence)
        if tree is None:
            records.append({
                "sample_id": "SYN{:04d}".format(index),
                "query": sentence,
                "parse_status": "UNPARSED",
                "parse_tree": "()",
                "intent": "UNKNOWN",
                "entities": [],
                "expected_chunk_ids": [],
                "expected_answerability": "REVIEW",
                "label_status": "UNREVIEWED",
            })
            continue

        intent, entities = semantic_labels(tree, aliases)
        expected = expected_chunks(intent, entities, knowledge_base.documents)
        answerability = "INSUFFICIENT_CONTEXT" if intent == "GET_DEADLINE" else (
            "EXPECTED_CONTEXT" if expected else "REVIEW_EXPECTED_CONTEXT"
        )
        records.append({
            "sample_id": "SYN{:04d}".format(index),
            "query": sentence,
            "parse_status": "PARSED",
            "parse_tree": tree.to_bracketed(),
            "intent": intent,
            "entities": entities,
            "expected_chunk_ids": expected,
            "expected_answerability": answerability,
            "label_status": "AUTO_LABEL_REQUIRES_REVIEW",
        })
    return records


def main():
    app_dir = Path(__file__).resolve().parent
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(description="Build labeled synthetic Course Assistant samples")
    parser.add_argument("--grammar", default=str(app_dir / "output" / "grammar.txt"))
    parser.add_argument("--data-dir", default=str(data_dir))
    parser.add_argument("--output-jsonl", default=str(app_dir / "output" / "samples_labeled.jsonl"))
    parser.add_argument("--output-text", default=str(app_dir / "output" / "samples.txt"))
    parser.add_argument("--max-sentences", type=int, default=10000)
    args = parser.parse_args()

    records = build_records(args.grammar, args.data_dir, args.max_sentences)
    jsonl_path, text_path = Path(args.output_jsonl), Path(args.output_text)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with text_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(record["query"] + "\n")
    parsed = sum(record["parse_status"] == "PARSED" for record in records)
    print("Wrote {} samples ({} parsed) to {} and {}".format(
        len(records), parsed, jsonl_path, text_path
    ))


if __name__ == "__main__":
    main()
