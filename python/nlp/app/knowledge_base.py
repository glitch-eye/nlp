"""Load the repository's course data into source-aware retrieval chunks."""

import re
from pathlib import Path

from nlp.app.text_utils import normalize, slug


DATA_FILES = {
    "course_info.txt": "course_info",
    "schedule.txt": "schedule",
    "topics.txt": "topics",
    "assignments.txt": "assignments",
    "resources.txt": "resources",
    "regulations.txt": "regulations",
    "faq.txt": "faq",
}


_slug = slug


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
        else:
            entity_id, canonical = _slug(alias).upper().replace("-", "_"), value
        aliases.append((alias, entity_id, canonical))
        if canonical and canonical.casefold() != alias.casefold():
            aliases.append((canonical, entity_id, canonical))
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
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        return [(block, _section_name(block)) for block in blocks]

    pieces = [piece.strip() for piece in re.split(pattern, text, flags=re.MULTILINE) if piece.strip()]
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
    normalized = " " + normalize(text) + " "
    found = []
    for alias, entity_id, canonical in aliases:
        needle = " " + normalize(alias) + " "
        if needle.strip() and needle in normalized and entity_id not in found:
            found.append(entity_id)
    for pattern, formatter in (
        (r"\\bWEEK\\s+0?(\\d{1,2})\\b", lambda match: "WEEK_{:02d}".format(int(match.group(1)))),
        (r"\\bCHAPTER\\s+0?(\\d{1,2})\\b", lambda match: "CH{:02d}".format(int(match.group(1)))),
        (r"\\bL\\s*\\.?\\s*O\\s*\\.?\\s*(\\d+\\.\\d+)\\b", lambda match: "LO" + match.group(1)),
        (r"\\bmidterm\\b", lambda match: "MIDTERM"),
        (r"\\bfinal\\b", lambda match: "FINAL"),
        (r"\\bAI[_ ]usage\\b", lambda match: "AI_USAGE"),
        (r"\\bLLM[_ /-]?RAG\\b", lambda match: "LLM_RAG"),
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entity_id = formatter(match)
            if entity_id not in found:
                found.append(entity_id)
    normalized_lower = normalize(text)
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


_normalize = normalize


class KnowledgeBase:
    """Source documents split into retrievable chunks with stable metadata."""

    def __init__(self, data_dir, entity_file=None):
        self.data_dir = Path(data_dir).resolve()
        self.entity_file = Path(entity_file).resolve() if entity_file else self.data_dir / "entities.txt"
        if not self.data_dir.is_dir():
            raise FileNotFoundError("Course data directory does not exist: {}".format(self.data_dir))
        missing = [name for name in DATA_FILES if not (self.data_dir / name).is_file()]
        if not self.entity_file.is_file():
            missing.append(self.entity_file.name)
        if missing:
            raise FileNotFoundError(
                "Incomplete course dataset at {}. Missing: {}".format(
                    self.data_dir, ", ".join(missing)
                )
            )
        self.aliases = load_entity_aliases(self.entity_file)
        self.documents = self._load_documents()
        if not self.documents:
            raise ValueError("Course dataset contains no retrievable documents: {}".format(self.data_dir))

    def _load_documents(self):
        documents = []
        for filename, kind in DATA_FILES.items():
            path = self.data_dir / filename
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
    """Return the course dataset bundled inside this repository (python/data)."""
    app_dir = Path(__file__).resolve().parent
    data_dir = app_dir.parents[1] / "data"
    required = list(DATA_FILES) + ["entities.txt"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Repository course dataset is incomplete at {}. Missing: {}. "
            "Restore the files under python/data or pass --data-dir explicitly.".format(
                data_dir, ", ".join(missing)
            )
        )
    return data_dir
