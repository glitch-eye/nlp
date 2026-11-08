"""Build lightweight semantic frames from parsed queries and KB chunks."""

import re
import unicodedata

from nlp.app.parser import CourseParser, ParseNode


TREE_INTENT_TO_INTENT = {
    "Q_COURSE_INFO": "GET_COURSE_INFO",
    "Q_SCHEDULE": "GET_SCHEDULE",
    "Q_TOPIC": "GET_TOPIC",
    "Q_ASSIGNMENT": "GET_ASSIGNMENT",
    "Q_DEADLINE": "GET_DEADLINE",
    "Q_RESOURCE": "GET_RESOURCE",
    "Q_RULE": "GET_RULE",
    "Q_LO": "GET_LO",
    "Q_UNKNOWN": "UNKNOWN",
}

KIND_TO_INTENTS = {
    "course_info": {"GET_COURSE_INFO", "GET_LO"},
    "schedule": {"GET_SCHEDULE"},
    "topics": {"GET_TOPIC"},
    "assignments": {"GET_ASSIGNMENT", "GET_DEADLINE"},
    "resources": {"GET_RESOURCE"},
    "regulations": {"GET_RULE"},
    "faq": {"GET_COURSE_INFO", "GET_SCHEDULE", "GET_TOPIC", "GET_ASSIGNMENT",
             "GET_DEADLINE", "GET_RESOURCE", "GET_RULE", "GET_LO"},
}


def _normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _walk(node):
    yield node
    for child in node.children:
        if isinstance(child, ParseNode):
            yield from _walk(child)


def _node_text(node):
    return " ".join(child.label for child in _walk(node) if not child.children).strip()


def _first_node(root, label):
    return next((node for node in _walk(root) if node.label == label), None)


def _find_alias_entities(text, aliases):
    normalized_text = " " + _normalize(text) + " "
    found = []
    for alias, entity_id, canonical in aliases:
        for phrase in (alias, canonical):
            normalized_phrase = " " + _normalize(phrase) + " "
            if normalized_phrase.strip() and normalized_phrase in normalized_text:
                if entity_id not in found:
                    found.append(entity_id)
    return found


def query_frame(query, parser, aliases=()):
    """Parse a query and derive its intent/entities; return None on parse failure."""
    try:
        tree = parser.parse(query)
    except Exception:
        # Retrieval must remain available if the CFG is temporarily malformed.
        return None
    if tree is None:
        return None

    return frame_from_tree(tree, query, aliases)


def frame_from_tree(tree, query, aliases=()):
    """Derive a semantic frame from an already parsed query tree."""
    root_question = next(
        (node for node in tree.children
         if isinstance(node, ParseNode) and node.label.startswith("Q_")),
        None,
    )
    intent = TREE_INTENT_TO_INTENT.get(root_question.label, "UNKNOWN") if root_question else "UNKNOWN"
    entities = []
    labels = (
        "COURSE_REF", "EXAM_REF", "WEEK_REF", "CHAPTER_REF", "TOPIC_REF",
        "ASSIGNMENT_REF", "PART_REF", "RESOURCE_TOPIC", "RESOURCE_REF",
        "RULE_REF", "RULE_OBJECT", "LO_REF", "DEADLINE_REF",
    )
    for label in labels:
        node = _first_node(tree, label)
        if node is None:
            continue
        value = _node_text(node)
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
            if intent in ("GET_ASSIGNMENT", "GET_DEADLINE"):
                entities.append("BTL01")
        elif label in ("RULE_OBJECT", "RULE_REF"):
            entities.append("LLM_RAG" if "llm" in value.casefold() else "AI_USAGE")
        elif label == "RESOURCE_TOPIC":
            normalized = _normalize(value)
            entities.append(
                "PYTHON_NLP" if "python" in normalized else
                "STATISTICAL_NLP" if "statistical" in normalized or "thong ke" in normalized else
                "MACHINE_TRANSLATION"
            )
        else:
            entities.extend(_find_alias_entities(value, aliases))

    # Structure-aware extraction covers numbered sections even when the grammar
    # represents the number as a terminal rather than a named semantic node.
    normalized_query = _normalize(query)
    for pattern, formatter in (
        (r"\b(?:tuan|week)\s+(?:thu\s+)?0?(\d{1,2})\b", lambda n: "WEEK_{:02d}".format(int(n))),
        (r"\b(?:chuong|chapter)\s+0?(\d{1,2})\b", lambda n: "CH{:02d}".format(int(n))),
    ):
        match = re.search(pattern, normalized_query)
        if match:
            entities.append(formatter(match.group(1)))

    return {"intent": intent, "entity_ids": list(dict.fromkeys(entities)),
            "parse_status": "PARSED"}


def chunk_frame(document):
    """Create the indexing-time frame for a chunk after structural chunking."""
    return {
        "intents": sorted(KIND_TO_INTENTS.get(document.get("kind"), set())),
        "entity_ids": list(document.get("entity_ids", [])),
        "kind": document.get("kind"),
    }
