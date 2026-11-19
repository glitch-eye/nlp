"""Safe clause segmentation for multi-intent course queries.

The base CFG intentionally remains a single-intent grammar.  A conjunction is
only treated as a clause boundary when every resulting segment independently
parses with that CFG.  This prevents splitting lexical coordination such as
"CFG và Parsing" or comparisons such as "Chương 3 và Chương 4".
"""
import re


_CONJUNCTION_RE = re.compile(r"\s+và\s+", re.IGNORECASE)


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
    """Return the best independently parseable clause partition.

    A dynamic-programming search allows "và" to remain inside a clause when it
    is lexical coordination (for example "CFG và Parsing"), while splitting
    other conjunctions into separate intents.  The original query is retained
    if no multi-clause partition is validated by the existing CFG.
    """
    text = (query or "").strip()
    if not text or parser is None:
        return [text] if text else []

    raw_parts = _CONJUNCTION_RE.split(text)
    if len(raw_parts) < 2:
        return [text]

    suffix = "?" if text.endswith("?") else (
        "!" if text.endswith("!") else ("." if text.endswith(".") else "")
    )
    count = len(raw_parts)
    memo = {}

    def best_from(index):
        if index == count:
            return []
        if index in memo:
            return memo[index]

        best = None
        for end in range(index + 1, count + 1):
            candidate = " và ".join(part.strip() for part in raw_parts[index:end]).strip()
            candidate = candidate.rstrip("?!.,;:")
            variants = [candidate + suffix] if suffix else [candidate, candidate + "?"]
            validated = next(
                (variant for variant in variants if variant and parser.parse(variant) is not None),
                None,
            )
            if validated is None:
                continue
            tail = best_from(end)
            if tail is None:
                continue
            option = [validated] + tail
            if best is None or len(option) > len(best):
                best = option

        memo[index] = best
        return best

    partition = best_from(0)
    return partition if partition and len(partition) > 1 else [text]


def is_multi_intent_query(query, parser):
    return len(split_intent_clauses(query, parser)) > 1
