"""TF-IDF retrieval for Course Assistant knowledge chunks."""

import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _entity_token(entity_id):
    """Keep a normalized entity ID as one distinctive TF-IDF token."""
    return "entityid" + re.sub(r"[^a-z0-9]", "", str(entity_id).casefold())


class Retriever:
    """Fit a TF-IDF index once and return ranked source-aware chunks."""

    def __init__(self, documents, aliases=None, top_k=3, min_score=0.08):
        self.documents = list(documents)
        self.aliases = aliases or []
        self.top_k = top_k
        # Initial threshold for inspection; calibrate against sample_queries.txt.
        self.min_score = min_score
        if not self.documents:
            raise ValueError("Retriever requires at least one document")

        corpus = []
        for document in self.documents:
            entity_text = " ".join(
                _entity_token(entity_id) for entity_id in document.get("entity_ids", [])
            )
            corpus.append("{} {} {}".format(document["text"], entity_text, entity_text))
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def expand_query(self, query, semantic=None):
        """Add matched lexicon aliases and semantic entity hints to the query."""
        additions = []
        normalized_query = " " + _normalize(query) + " "
        for alias, entity_id, canonical in self.aliases:
            normalized_alias = " " + _normalize(alias) + " "
            if normalized_alias.strip() and normalized_alias in normalized_query:
                additions.extend([canonical, _entity_token(entity_id)])

        # Resolve explicit identifiers even when the entity lexicon uses a
        # different surface form, such as Vietnamese week numbers in schedule.txt.
        for pattern, formatter in (
            (r"\b(?:tuần|week)\s+(?:thứ\s+)?0?(\d{1,2})\b",
             lambda match: "WEEK_{:02d}".format(int(match.group(1)))),
            (r"\b(?:chương|chapter)\s+0?(\d{1,2})\b",
             lambda match: "CH{:02d}".format(int(match.group(1)))),
            (r"\bL\s*\.?\s*O\s*\.?\s*(\d+\.\d+)\b",
             lambda match: "LO" + match.group(1)),
        ):
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                additions.append(_entity_token(formatter(match)))
        if re.search(r"\bC03085\b", query, re.IGNORECASE):
            additions.append(_entity_token("CO3085"))

        if semantic:
            entity_ids = list(semantic.get("entity_ids", [])) if isinstance(semantic, dict) else []
            if isinstance(semantic, dict):
                entity_ids += list(semantic.get("entities", {}).values()) if isinstance(semantic.get("entities"), dict) else []
            additions.extend(_entity_token(value) for value in entity_ids)
            intent_terms = {
                "GET_COURSE_INFO": "course information credits exams",
                "GET_SCHEDULE": "schedule week weekly plan lesson topics",
                "GET_TOPIC": "chapter topic summary subtopics",
                "GET_ASSIGNMENT": "assignment project requirements parts",
                "GET_DEADLINE": "deadline due date submission",
                "GET_RESOURCE": "resources textbook reference books",
                "GET_RULE": "regulations rules policies",
                "GET_LO": "learning outcomes LO",
            }
            intent = semantic.get("intent") if isinstance(semantic, dict) else None
            if intent in intent_terms:
                additions.append(intent_terms[intent])

        if not additions:
            return query
        return "{} {}".format(query, " ".join(dict.fromkeys(additions)))

    def retrieve(self, question, semantic=None, top_k=None):
        query = self.expand_query(question, semantic=semantic)
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        limit = self.top_k if top_k is None else top_k
        selected = [(index, float(score)) for index, score in ranked[:limit] if score >= self.min_score]
        return {
            "query": question,
            "chunks": [self.documents[index] for index, _ in selected],
            "scores": [score for _, score in selected],
            "retrieval_status": "FOUND" if selected else "NO_RELEVANT_CONTEXT",
        }


def format_trace(result):
    """Render one retrieval result for the human-readable retrieval trace."""
    lines = [
        "Query: {}".format(result["query"]),
        "Retrieval status: {}".format(result["retrieval_status"]),
    ]
    if not result["chunks"]:
        lines.append("No chunk met the relevance threshold.")
    for rank, (chunk, score) in enumerate(zip(result["chunks"], result["scores"]), 1):
        lines.extend([
            "{}. {} | score={:.4f} | {} / {}".format(
                rank, chunk["chunk_id"], score, chunk["source"], chunk["section"]
            ),
            "   entity_ids: {}".format(", ".join(chunk.get("entity_ids", [])) or "(none)"),
            "   {}".format(" ".join(chunk["text"].split())),
        ])
    return "\n".join(lines)
