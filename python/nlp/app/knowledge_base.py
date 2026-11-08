"""Load the sample course files into source-aware retrieval chunks."""

import re
import unicodedata
from pathlib import Path


DATA_FILES = {
    "course_info.txt": "course_info",
    "schedule.txt": "schedule",
    "topics.txt": "topics",
    "assignments.txt": "assignments",
    "resources.txt": "resources",
    "regulations.txt": "regulations",
    "faq.txt": "faq",
}


def _slug(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "section"


def _read_text(path):
    return Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def load_entity_aliases(path):
    """Return ordered (alias, entity_id, canonical_value) entries."""
    aliases = []
    current_group = ""
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("="):
            continue
        if "=" not in line:
            current_group = line.rstrip(":").casefold()
            continue
        alias, value = [part.strip() for part in line.split("=", 1)]
        if current_group == "chapters":
            entity_id, canonical = value, alias
        elif current_group == "topics":
            entity_id, canonical = _slug(alias).upper().replace("-", "_"), value
        elif current_group == "assignment":
            entity_id, canonical = _slug(alias).upper().replace("-", "_"), value
        else:
            entity_id, canonical = _slug(alias).upper().replace("-", "_"), value
        aliases.append((alias, entity_id, canonical))
        if canonical and canonical.casefold() != alias.casefold():
            aliases.append((canonical, entity_id, canonical))
    # Longer phrases first so a specific alias wins over a substring.
    return sorted(aliases, key=lambda item: len(item[0]), reverse=True)


def _sections(text, kind):
    """Split a source into semantic sections while preserving its content."""
    if kind == "schedule":
        pattern = r"(?=^WEEK\s+\d+\b)"
    elif kind == "topics":
        pattern = r"(?=^CHAPTER\s+\d+\s*:)"
    elif kind == "faq":
        pattern = r"(?=^Q\d+\s*:)"
    elif kind == "resources":
        pattern = r"(?=^(?:TEXTBOOK\s+\d+|REFERENCE\s+\d+|study_note\s*:))"
    elif kind == "assignments":
        pattern = r"(?=^(?:PART\s+[IVX]+\s*-|LLM/RAG EXTENSION))"
    elif kind == "regulations":
        pattern = r"(?=^(?:attendance|academic_integrity|AI_usage|source_code|assignment_sharing|external_libraries|LLM_RAG|OUT_OF_SCOPE)\s*:)"
    else:
        # Field and paragraph boundaries keep unrelated facts apart.
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        return [(block, _section_name(block)) for block in blocks]

    pieces = [piece.strip() for piece in re.split(pattern, text, flags=re.MULTILINE) if piece.strip()]
    # Strip a title banner while retaining any explanatory preamble below it.
    if pieces and re.match(r"^[A-Z][A-Z0-9 /&_-]+\n=+", pieces[0]):
        first_lines = pieces[0].splitlines()
        preamble = "\n".join(first_lines[2:]).strip()
        pieces = ([preamble] if preamble else []) + pieces[1:]
    return [(piece, _section_name(piece)) for piece in pieces]


def _section_name(text):
    for line in text.splitlines():
        line = line.strip().strip("=# ")
        if not line:
            continue
        if line.endswith(":"):
            return line[:-1].strip()
        if re.match(r"^(WEEK\s+\d+|CHAPTER\s+\d+|Q\d+|PART\s+[IVX]+|TEXTBOOK\s+\d+|REFERENCE\s+\d+)", line, re.I):
            return line
        return line[:100]
    return "document"


def _entities_in(text, aliases):
    normalized = " " + _normalize(text) + " "
    found = []
    for alias, entity_id, canonical in aliases:
        needle = " " + _normalize(alias) + " "
        if needle.strip() and needle in normalized and entity_id not in found:
            found.append(entity_id)
    for pattern, formatter in (
        (r"\bWEEK\s+0?(\d{1,2})\b", lambda match: "WEEK_{:02d}".format(int(match.group(1)))),
        (r"\bCHAPTER\s+0?(\d{1,2})\b", lambda match: "CH{:02d}".format(int(match.group(1)))),
        (r"\bL\s*\.?\s*O\s*\.?\s*(\d+\.\d+)\b", lambda match: "LO" + match.group(1)),
        (r"\bmidterm\b", lambda match: "MIDTERM"),
        (r"\bfinal\b", lambda match: "FINAL"),
        (r"\bAI[_ ]usage\b", lambda match: "AI_USAGE"),
        (r"\bLLM[_ /-]?RAG\b", lambda match: "LLM_RAG"),
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entity_id = formatter(match)
            if entity_id not in found:
                found.append(entity_id)
    normalized_lower = _normalize(text)
    resource_aliases = (
        ("PYTHON_NLP", "python" in normalized_lower and
         ("nlp" in normalized_lower or "natural language processing" in normalized_lower)),
        ("STATISTICAL_NLP", "statistical" in normalized_lower and
         ("nlp" in normalized_lower or "natural language processing" in normalized_lower)),
        ("MACHINE_TRANSLATION", "machine translation" in normalized_lower),
    )
    for entity_id, matched in resource_aliases:
        if matched and entity_id not in found:
            found.append(entity_id)
    return found


def _normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


class KnowledgeBase:
    """Source documents split into retrievable chunks with stable metadata."""

    def __init__(self, data_dir, entity_file=None):
        self.data_dir = Path(data_dir)
        self.entity_file = Path(entity_file) if entity_file else self.data_dir / "entities.txt"
        self.aliases = load_entity_aliases(self.entity_file) if self.entity_file.exists() else []
        self.documents = self._load_documents()

    def _load_documents(self):
        documents = []
        for filename, kind in DATA_FILES.items():
            path = self.data_dir / filename
            if not path.exists():
                continue
            for index, (text, section) in enumerate(_sections(_read_text(path), kind)):
                if not text.strip():
                    continue
                documents.append({
                    "chunk_id": "{}-{}-{}".format(_slug(Path(filename).stem), _slug(section), index),
                    "text": text,
                    "source": filename,
                    "section": section,
                    "entity_ids": _entities_in(text, self.aliases),
                    "kind": kind,
                })
        return documents


def default_data_dir():
    """Find the sample dataset from the repository's standard layout."""
    app_dir = Path(__file__).resolve().parent
    candidates = [
        app_dir.parents[4] / "NLP_Course_Assistant_Sample_Data",
        app_dir.parents[3] / "NLP_Course_Assistant_Sample_Data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find NLP_Course_Assistant_Sample_Data; pass data_dir explicitly")
