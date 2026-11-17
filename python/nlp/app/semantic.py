"""Semantic adapter (Member 2): ParseNode -> intent / entity frame.

Hook contract used by ``assistant.py``::

    interpret(parse_tree, question) -> {"intent", "entities", "entity_ids", "parse_status"}

The tree walk itself lives in ``semantic_frame.frame_from_tree`` (Part I output);
this module adds what the RAG layer needs on top of it:

* alias lookup from ``entities.txt`` so TOPIC_REF/RESOURCE_REF nodes get canonical IDs,
* a category-keyed ``entities`` dict (``{"chapter": "CH04"}``) for dialogue state,
* an honest ``UNPARSED`` frame when the parser produced no tree (no guessed intent).

The frame is a retrieval *hint*; retrieved chunks remain the only evidence.
"""

import re

from nlp.app.semantic_frame import frame_from_tree

UNPARSED = "UNPARSED"
PARSED = "PARSED"

_ALIAS_CACHE = {}


def entity_category(entity_id):
    """Map a canonical entity ID to a slot name (mirrors dialogue.entity_category)."""
    if not entity_id:
        return None
    if re.match(r"^CH\d{2}$", entity_id):
        return "chapter"
    if re.match(r"^WEEK_\d{2}$", entity_id):
        return "week"
    if entity_id in ("PART_I", "PART_II", "PART_III", "PART_IV"):
        return "part"
    if entity_id == "BTL01":
        return "assignment"
    if entity_id in ("CO3085", "NLP", "NATURAL_LANGUAGE_PROCESSING"):
        return "course"
    if entity_id.startswith("LO"):
        return "lo"
    if entity_id in ("MIDTERM", "FINAL"):
        return "exam"
    if entity_id in ("AI_USAGE", "LLM_RAG"):
        return "rule"
    return "topic"


def load_aliases(data_dir=None):
    """Load (alias, entity_id, canonical) entries from entities.txt; [] if unavailable.

    Cached per directory. Failure to find the data is not fatal: the parser's own
    entity extraction still works, only alias-based topics are lost.
    """
    key = str(data_dir) if data_dir else ""
    if key in _ALIAS_CACHE:
        return _ALIAS_CACHE[key]
    aliases = []
    try:
        from nlp.app.knowledge_base import default_data_dir, load_entity_aliases
        from pathlib import Path

        base = Path(data_dir) if data_dir else Path(default_data_dir())
        entity_file = base / "entities.txt"
        if entity_file.exists():
            aliases = load_entity_aliases(entity_file)
    except Exception:
        aliases = []
    _ALIAS_CACHE[key] = aliases
    return aliases


def unparsed_frame():
    """Frame for questions the grammar did not accept. Deliberately has no intent."""
    return {"intent": None, "entities": {}, "entity_ids": [], "parse_status": UNPARSED}


def interpret(parse_tree, question, aliases=None):
    """Turn a Part I parse tree into the standard semantic frame.

    ``parse_tree`` may be ``None`` (parser rejected the sentence); the result is then
    the UNPARSED frame, so callers can still retrieve on the raw question.
    """
    if parse_tree is None:
        return unparsed_frame()

    if aliases is None:
        aliases = load_aliases()
    base = frame_from_tree(parse_tree, question or "", aliases)

    entity_ids = list(base.get("entity_ids") or [])
    entities = {}
    for entity_id in entity_ids:
        category = entity_category(entity_id)
        # First hit per category wins, so the order of the question is respected.
        if category and category not in entities:
            entities[category] = entity_id

    return {
        "intent": base.get("intent"),
        "entities": entities,
        "entity_ids": entity_ids,
        "parse_status": PARSED,
    }


# Aliases kept so assistant.py's probing loop works whichever name it tries first.
extract_semantic_frame = interpret
