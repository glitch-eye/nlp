"""TF-IDF retrieval for Course Assistant knowledge chunks."""

import re
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.app.semantic_frame import chunk_frame, query_frame
from nlp.app.text_utils import normalize
from nlp.app.multi_intent import split_intent_clauses


_normalize = normalize


def _entity_token(entity_id):
    """Keep a normalized entity ID as one distinctive TF-IDF token."""
    return "entityid" + re.sub(r"[^a-z0-9]", "", str(entity_id).casefold())



def _infer_schedule_week(document):
    """Infer a WEEK_XX entity from an explicit schedule heading/section."""
    if document.get("kind") != "schedule":
        return None
    candidates = [str(document.get("section", "")), str(document.get("text", ""))[:160]]
    for candidate in candidates:
        match = re.search(r"\b(?:week|tuần)\s+(?:thứ\s+)?0?(\d{1,2})\b",
                         candidate, re.IGNORECASE)
        if match:
            return "WEEK_{:02d}".format(int(match.group(1)))
    return None



def _infer_schedule_chapter(document):
    """Infer a CHXX entity from chapter metadata in schedule chunks."""
    if document.get("kind") != "schedule":
        return None
    candidates = [str(document.get("section", "")), str(document.get("text", ""))[:240]]
    for candidate in candidates:
        match = re.search(r"\bchapter\s+0?(\d{1,2})\b", candidate, re.IGNORECASE)
        if match:
            return "CH{:02d}".format(int(match.group(1)))
    return None



def _comparison_entity_count(query):
    """Count explicit chapter/week references in a comparison-style query."""
    comparison = re.search(
        r"\b(compare|comparison|compared|different|difference|differences|similar|contrast|khác nhau|so sánh|giống nhau)\b",
        query, re.IGNORECASE,
    )
    if not comparison:
        return 0
    refs = re.findall(r"\b(?:chapter|chương|week|tuần)\s+(?:thứ\s+)?0?\d{1,2}\b",
                      query, re.IGNORECASE)
    return len(refs)


class Retriever:
    """Fit a TF-IDF index once and return ranked source-aware chunks."""

    def __init__(self, documents, aliases=None, top_k=3, min_score=0.08,
                 parser=None, lexical_weight=0.75, min_lexical_score=0.0):
        self.documents = list(documents)
        # Schedule chunks may mention the week in their heading/section without
        # storing it as an entity. Add that ID before frames/index are built.
        for document in self.documents:
            entity_ids = document.setdefault("entity_ids", [])
            week_id = _infer_schedule_week(document)
            chapter_id = _infer_schedule_chapter(document)
            for inferred_id in (week_id, chapter_id):
                if inferred_id and inferred_id not in entity_ids:
                    entity_ids.append(inferred_id)
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

    def retrieve_multi(self, question, clauses, clause_frames, top_k=None):
        """Retrieve evidence for all intents in one batched, coverage-aware pass.

        Each clause is scored against the same fitted chunk index. Candidate chunks
        are globally deduplicated, then selected to maximize intent coverage while
        retaining relevance. No model/API call is made here.
        """
        if len(clauses) != len(clause_frames):
            raise ValueError("clauses and clause_frames must have equal length")
        if not clauses:
            return {
                "query": question, "chunks": [], "scores": [],
                "retrieval_status": "NO_RELEVANT_CONTEXT",
                "intent": "MULTI_INTENT", "clauses": [],
                "intent_coverage": 0.0, "score_details": [],
            }

        expanded_queries = [
            self.expand_query(clause, semantic=frame)
            for clause, frame in zip(clauses, clause_frames)
        ]
        query_matrix = self.vectorizer.transform(expanded_queries)
        lexical_matrix = cosine_similarity(query_matrix, self.matrix)

        # Build one candidate pool from a batched query-vs-all-chunks score matrix.
        candidates = {}
        per_clause = []
        pool_size = min(len(self.documents), max(8, self.top_k * 4))
        for clause_index, frame in enumerate(clause_frames):
            lexical_scores = lexical_matrix[clause_index]
            semantic_scores = self._semantic_scores(frame)
            if frame and frame.get("intent") not in (None, "UNKNOWN", "MULTI_INTENT"):
                combined = [
                    self.lexical_weight * float(lexical)
                    + self.semantic_weight * float(semantic)
                    for lexical, semantic in zip(lexical_scores, semantic_scores)
                ]
            else:
                combined = [float(value) for value in lexical_scores]

            ranked = sorted(
                range(len(self.documents)),
                key=lambda index: (-float(combined[index]), index),
            )
            accepted = []
            query_entities = set(frame.get("entity_ids", [])) if frame else set()
            for index in ranked:
                entity_match = query_entities.intersection(
                    self.chunk_frames[index]["entity_ids"]
                )
                if (combined[index] < self.min_score or
                    (lexical_scores[index] < self.min_lexical_score and not entity_match)):
                    continue
                accepted.append(index)
                entry = candidates.setdefault(index, {
                    "chunk": self.documents[index],
                    "scores_by_clause": {},
                    "details_by_clause": {},
                })
                entry["scores_by_clause"][clause_index] = float(combined[index])
                entry["details_by_clause"][clause_index] = {
                    "lexical": float(lexical_scores[index]),
                    "semantic": float(semantic_scores[index]),
                    "combined": float(combined[index]),
                }
                if len(accepted) >= pool_size:
                    break
            per_clause.append({
                "query": clauses[clause_index],
                "intent": (frame or {}).get("intent"),
                "entity_ids": list((frame or {}).get("entity_ids", [])),
                "candidate_chunk_ids": [
                    self.documents[index]["chunk_id"] for index in accepted
                ],
                "retrieval_status": "FOUND" if accepted else "NO_RELEVANT_CONTEXT",
            })

        # Greedy maximum-coverage selection with relevance as a tie-breaker.
        limit = self.top_k if top_k is None else top_k
        limit = max(int(limit), len(clauses))
        selected = []
        selected_indices = set()
        covered = set()
        while candidates and len(selected) < limit:
            best_index = None
            best_objective = None
            for index, candidate in candidates.items():
                if index in selected_indices:
                    continue
                support = set(candidate["scores_by_clause"])
                newly_covered = support - covered
                best_score = max(candidate["scores_by_clause"].values())
                objective = (
                    len(newly_covered) / float(len(clauses)) * 0.55
                    + best_score * 0.45
                )
                tie_break = (objective, best_score, -index)
                if best_objective is None or tie_break > best_objective:
                    best_objective = tie_break
                    best_index = index
            if best_index is None:
                break
            selected_indices.add(best_index)
            candidate = candidates[best_index]
            support = set(candidate["scores_by_clause"])
            covered.update(support)
            selected.append((best_index, candidate))

            # Once all clauses are covered, fill remaining slots only with
            # useful additional evidence; do not force exactly 3 chunks/intent.
            if len(covered) == len(clauses) and len(selected) >= min(limit, len(clauses)):
                remaining = [
                    item for item in candidates
                    if item not in selected_indices
                ]
                if not remaining:
                    break

        chunks = [item[1]["chunk"] for item in selected]
        scores = [
            max(item[1]["scores_by_clause"].values()) for item in selected
        ]
        details = []
        support_rows = []
        for _, candidate in selected:
            support_indices = sorted(candidate["scores_by_clause"])
            support_rows.append({
                "chunk_id": candidate["chunk"]["chunk_id"],
                "clause_indices": support_indices,
                "intents": [
                    (clause_frames[i] or {}).get("intent") for i in support_indices
                ],
            })
            details.append({
                "combined": max(candidate["scores_by_clause"].values()),
                "per_clause": candidate["details_by_clause"],
            })

        covered_clause_indices = set(
            index for item in selected
            for index in item[1]["scores_by_clause"]
        )
        return {
            "query": question,
            "chunks": chunks,
            "scores": scores,
            "retrieval_status": "FOUND" if chunks else "NO_RELEVANT_CONTEXT",
            "parse_status": "PARSED" if all(
                (frame or {}).get("parse_status") == "PARSED"
                for frame in clause_frames
            ) else "PARTIAL",
            "intent": "MULTI_INTENT",
            "intents": [(frame or {}).get("intent") for frame in clause_frames],
            "clauses": per_clause,
            "entity_ids": list(dict.fromkeys(
                entity_id for frame in clause_frames
                for entity_id in (frame or {}).get("entity_ids", [])
            )),
            "expanded_query": " AND ".join(expanded_queries),
            "score_details": details,
            "chunk_support": support_rows,
            "intent_coverage": len(covered_clause_indices) / float(len(clauses)),
            "covered_clause_indices": sorted(covered_clause_indices),
            "candidate_chunk_count": len(candidates),
            "selected_chunk_count": len(chunks),
        }

    def retrieve(self, question, semantic=None, top_k=None):
        clauses = split_intent_clauses(question, self.parser)
        if len(clauses) > 1 and semantic is None:
            child_results = [self.retrieve(clause, top_k=top_k) for clause in clauses]
            ranked = []
            seen = set()
            for clause_index, result in enumerate(child_results):
                for chunk, score, detail in zip(
                    result.get("chunks", []), result.get("scores", []),
                    result.get("score_details", []),
                ):
                    cid = chunk.get("chunk_id")
                    if cid in seen:
                        continue
                    seen.add(cid)
                    ranked.append((float(score), clause_index, chunk, detail))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            limit = self.top_k if top_k is None else top_k
            # Multi-intent questions need evidence for every clause. Allow up to
            # two chunks per clause (capped at six) while preserving top_k as the
            # default budget for ordinary single-intent queries.
            retrieval_budget = max(limit, min(6, 2 * len(child_results)))
            # Reserve one evidence slot per clause before filling remaining
            # slots globally, so a high-scoring clause cannot erase another.
            selected = []
            selected_keys = set()
            for clause_index in range(len(child_results)):
                candidate = next(
                    (item for item in ranked
                     if item[1] == clause_index and item[2].get("chunk_id") not in selected_keys),
                    None,
                )
                if candidate is not None and len(selected) < retrieval_budget:
                    selected.append(candidate)
                    selected_keys.add(candidate[2].get("chunk_id"))
            for item in ranked:
                if len(selected) >= retrieval_budget:
                    break
                cid = item[2].get("chunk_id")
                if cid not in selected_keys:
                    selected.append(item)
                    selected_keys.add(cid)
            return {
                "query": question,
                "chunks": [item[2] for item in selected],
                "scores": [item[0] for item in selected],
                "retrieval_status": "FOUND" if selected else "NO_RELEVANT_CONTEXT",
                "parse_status": "PARSED",
                "intent": "MULTI_INTENT",
                "intents": [item.get("intent") for item in child_results],
                "clauses": [
                    {"query": clause, "intent": item.get("intent"),
                     "entity_ids": item.get("entity_ids", []),
                     "retrieval_status": item.get("retrieval_status")}
                    for clause, item in zip(clauses, child_results)
                ],
                "entity_ids": list(dict.fromkeys(
                    eid for item in child_results for eid in item.get("entity_ids", [])
                )),
                "expanded_query": " AND ".join(item.get("expanded_query", clause)
                                                for item, clause in zip(child_results, clauses)),
                "score_details": [item[3] for item in selected],
            }
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
        comparison_refs = _comparison_entity_count(question)
        if comparison_refs >= 2:
            # Comparisons require evidence for both referenced entities. Keep the
            # default small, but expand the context budget for explicit compares.
            limit = max(limit, min(6, 2 * comparison_refs))
        if frame and frame.get("intent") == "UNKNOWN":
            selected = []
        else:
            # Filter before top_k: irrelevant chunks should not crowd out a
            # matching entity chunk at the ranking cutoff.
            eligible = [(index, float(score)) for index, score in ranked
                       if score >= self.min_score and
                       (lexical_scores[index] >= self.min_lexical_score or
                        query_entities.intersection(self.chunk_frames[index]["entity_ids"]))]
            selected = eligible[:limit]
        details = [{"lexical": float(lexical_scores[index]),
                    "semantic": float(semantic_scores[index]), "combined": score,
                    "entity_match": bool(query_entities.intersection(
                        self.chunk_frames[index]["entity_ids"]))}
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
