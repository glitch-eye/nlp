"""TF-IDF retrieval for Course Assistant knowledge chunks."""

import re
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.app.semantic_frame import chunk_frame, query_frame
from nlp.app.text_utils import normalize


_normalize = normalize


def _entity_token(entity_id):
    """Keep a normalized entity ID as one distinctive TF-IDF token."""
    return "entityid" + re.sub(r"[^a-z0-9]", "", str(entity_id).casefold())


class Retriever:
    """Fit a TF-IDF index once and return ranked source-aware chunks."""

    def __init__(self, documents, aliases=None, top_k=3, min_score=0.08,
                 parser=None, lexical_weight=0.75, min_lexical_score=0.0):
        self.documents = list(documents)
        self.aliases = aliases or []
        self.top_k = top_k
        # Initial threshold for inspection; calibrate against sample_queries.txt.
        self.min_score = min_score
        self.min_lexical_score = min_lexical_score
        self.lexical_weight = lexical_weight
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1")
        self.semantic_weight = 1.0 - lexical_weight
        self.chunk_frames = [chunk_frame(document) for document in self.documents]
        if parser is None:
            try:
                from nlp.app.parser import CourseParser
                parser = CourseParser()
            except Exception as error:
                # A broken/missing grammar degrades to lexical-only retrieval.
                logging.warning("Could not load semantic parser; using lexical retrieval: %s", error)
                parser = None
        self.parser = parser
        if not self.documents:
            raise ValueError("Retriever requires at least one document")

        corpus = []
        for document in self.documents:
            entity_text = " ".join(
                _entity_token(entity_id) for entity_id in document.get("entity_ids", [])
            )
            corpus.append("{} {} {}".format(document["text"], entity_text, entity_text))
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize,
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

    def _semantic_scores(self, frame):
        if not frame or frame.get("intent") in (None, "UNKNOWN"):
            return [0.0] * len(self.documents)
        intent = frame.get("intent")
        query_entities = set(frame.get("entity_ids", []))
        scores = []
        for chunk in self.chunk_frames:
            intent_match = intent in chunk["intents"]
            # FAQ chunks can support many intents, but should not outrank the
            # primary source category for a broad, entity-free question.
            faq_fallback = chunk["kind"] == "faq" and intent_match
            entity_match = query_entities.intersection(chunk["entity_ids"])
            # Entity agreement is the stronger signal when entities exist.
            if not query_entities:
                # Generic questions use parsed intent to select their source kind.
                scores.append(0.1 if faq_fallback else (1.0 if intent_match else 0.0))
                continue
            entity_score = len(entity_match) / float(len(query_entities))
            intent_prior = 0.05 if faq_fallback else (0.2 if intent_match else 0.0)
            scores.append(intent_prior + 0.8 * entity_score)
        return scores

    def retrieve(self, question, semantic=None, top_k=None):
        parsed_frame = query_frame(question, self.parser, self.aliases) if self.parser else None
        frame = semantic or parsed_frame
        query = self.expand_query(question, semantic=semantic)
        query_entities = set((frame or {}).get("entity_ids", []))
        normalized_query = " " + _normalize(question) + " "
        for alias, entity_id, canonical in self.aliases:
            normalized_alias = " " + _normalize(alias) + " "
            if normalized_alias.strip() and normalized_alias in normalized_query:
                query_entities.add(entity_id)
        if " bai tap lon " in normalized_query:
            query_entities.add("BTL01")
            query += " " + _entity_token("BTL01")
        query_vector = self.vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_vector, self.matrix).ravel()
        semantic_scores = self._semantic_scores(frame)
        if frame and frame.get("intent") not in (None, "UNKNOWN"):
            scores = [
                self.lexical_weight * float(lexical) + self.semantic_weight * semantic_score
                for lexical, semantic_score in zip(lexical_scores, semantic_scores)
            ]
        else:
            scores = [float(score) for score in lexical_scores]
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        limit = self.top_k if top_k is None else top_k
        if frame and frame.get("intent") == "UNKNOWN":
            selected = []
        else:
            selected = [(index, float(score)) for index, score in ranked[:limit]
                        if score >= self.min_score and
                        (lexical_scores[index] >= self.min_lexical_score or
                         query_entities.intersection(self.chunk_frames[index]["entity_ids"]))]
        details = [{"lexical": float(lexical_scores[index]),
                    "semantic": float(semantic_scores[index]), "combined": score}
                   for index, score in selected]
        parse_status = ((frame or {}).get("parse_status") or
                        ("PARSER_UNAVAILABLE" if self.parser is None else
                         ("PARSED" if parsed_frame else "UNPARSED")))
        return {
            "query": question,
            "chunks": [self.documents[index] for index, _ in selected],
            "scores": [score for _, score in selected],
            "retrieval_status": "FOUND" if selected else "NO_RELEVANT_CONTEXT",
            "parse_status": parse_status,
            "intent": (frame or {}).get("intent"),
            "entity_ids": sorted(query_entities),
            "expanded_query": query,
            "score_details": details,
        }


def format_trace(result):
    """Render one retrieval result for the human-readable retrieval trace."""
    lines = [
        "Query: {}".format(result["query"]),
        "Retrieval status: {}".format(result["retrieval_status"]),
        "Parse status: {}".format(result.get("parse_status", "UNKNOWN")),
        "Intent: {}".format(result.get("intent") or "(none)"),
        "Query entities: {}".format(", ".join(result.get("entity_ids", [])) or "(none)"),
        "Expanded query: {}".format(result.get("expanded_query", result["query"])),
    ]
    if not result["chunks"]:
        lines.append("No chunk met the relevance threshold.")
    for rank, (chunk, score) in enumerate(zip(result["chunks"], result["scores"]), 1):
        detail = result.get("score_details", [{}] * len(result["chunks"]))[rank - 1]
        lines.extend([
            "{}. {} | score={:.4f} | {} / {}".format(
                rank, chunk["chunk_id"], score, chunk["source"], chunk["section"]
            ),
            "   entity_ids: {}".format(", ".join(chunk.get("entity_ids", [])) or "(none)"),
            "   lexical={:.4f} semantic={:.4f} combined={:.4f}".format(detail.get("lexical", 0), detail.get("semantic", 0), detail.get("combined", score)),
            "   {}".format(" ".join(chunk["text"].split())),
        ])
    return "\n".join(lines)
