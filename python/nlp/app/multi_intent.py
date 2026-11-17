"""Safe clause segmentation for multi-intent course queries.

The base CFG intentionally remains a single-intent grammar.  A conjunction is
only treated as a clause boundary when every resulting segment independently
parses with that CFG.  This prevents splitting lexical coordination such as
"CFG và Parsing" or comparisons such as "Chương 3 và Chương 4".
"""
import re


_CONJUNCTION_RE = re.compile(r"\\s+và\\s+", re.IGNORECASE)


def _with_sentence_punctuation(parts, original):
    suffix = "?" if original.rstrip().endswith("?") else (
        "!" if original.rstrip().endswith("!") else (
            "." if original.rstrip().endswith(".") else ""
        )
    )
    output = []
    for part in parts:
        part = part.strip().rstrip("?!.,;:")
        if suffix and part:
            part += suffix
        output.append(part)
    return output


def split_intent_clauses(query, parser):
    """Return independently parseable clauses, or [query] when no safe split exists.

    This deliberately uses the existing parser as a validation oracle rather
    than guessing that every occurrence of "và" means multiple intents.
    """
    text = (query or "").strip()
    if not text or parser is None:
        return [text] if text else []

    raw_parts = _CONJUNCTION_RE.split(text)
    if len(raw_parts) < 2:
        return [text]

    parts = _with_sentence_punctuation(raw_parts, text)
    if len(parts) < 2 or any(not part for part in parts):
        return [text]

    # Every clause must be valid on its own.  This also protects coordinated
    # entities and topic names that belong to one intent.
    if all(parser.parse(part) is not None for part in parts):
        return parts
    return [text]


def is_multi_intent_query(query, parser):
    return len(split_intent_clauses(query, parser)) > 1
