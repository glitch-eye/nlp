"""TF-IDF retrieval for Course Assistant knowledge chunks."""

import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


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
            entity_text = " ".join(document.get("entity_ids", []))
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
                additions.extend([canonical, entity_id])

        if semantic:
            entity_ids = list(semantic.get("entity_ids", [])) if isinstance(semantic, dict) else []
            if isinstance(semantic, dict):
                entity_ids += list(semantic.get("entities", {}).values()) if isinstance(semantic.get("entities"), dict) else []
            additions.extend(str(value) for value in entity_ids)

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
