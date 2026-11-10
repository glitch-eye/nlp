from copy import deepcopy
import re

from nlp.app.text_utils import normalize


# Canonical display names aligned with output/grammar.txt so rewritten queries parse cleanly.
ENTITY_DISPLAY = {
    "CO3085": "môn học CO3085",
    "NLP": "Môn NLP",
    "NATURAL_LANGUAGE_PROCESSING": "Xử lý Ngôn ngữ Tự nhiên",
    "BTL01": "bài tập lớn",
    "PART_I": "Phần I",
    "PART_II": "Phần II",
    "PART_III": "Phần III",
    "PART_IV": "Phần IV",
    "SYNTACTIC_PROCESSING": "Phần I",
    "SEMANTIC_INTERPRETATION": "Phần II",
    "CONTEXT_AND_WORLD_KNOWLEDGE": "Phần III",
    "CH01": "Chương 1",
    "CH02": "Chương 2",
    "CH03": "Chương 3",
    "CH04": "Chương 4",
    "CH05": "Chương 5",
    "CH06": "Chương 6",
    "CH07": "Chương 7",
    "CH08": "Chương 8",
    "CH09": "Chương 9",
    "CH10": "Chương 10",
    "CH11": "Chương 11",
    "CH12": "Chương 12",
    "WEEK_01": "Tuần 1",
    "WEEK_02": "Tuần 2",
    "WEEK_03": "Tuần 3",
    "WEEK_04": "Tuần 4",
    "WEEK_05": "Tuần 5",
    "WEEK_06": "Tuần 6",
    "WEEK_07": "Tuần 7",
    "WEEK_08": "Tuần 8",
    "WEEK_09": "Tuần 9",
    "WEEK_10": "Tuần 10",
    "WEEK_11": "Tuần 11",
    "WEEK_12": "Tuần 12",
    "WEEK_13": "Tuần 13",
    "WEEK_14": "Tuần 14",
    "WEEK_15": "Tuần 15",
    "CFG": "CFG",
    "PARSING": "Parsing",
    "PCFG": "PCFG",
    "DEPENDENCY_GRAMMAR": "văn phạm phụ thuộc",
    "FEATURES": "nét và văn phạm gia tố",
    "AUGMENTED_GRAMMAR": "nét và văn phạm gia tố",
    "UNIFICATION": "nét và văn phạm gia tố",
    "POS_TAGGING": "gán nhãn từ loại",
    "LOGICAL_FORM": "dạng luận lý",
    "THEMATIC_ROLES": "vai trò chủ đề",
    "SELECTIONAL_RESTRICTION": "giới hạn lựa chọn",
    "SEMANTIC_NETWORK": "mạng ngữ nghĩa",
    "WSD": "giải nhập nhằng nghĩa từ",
    "SEMANTIC_GRAMMAR": "văn phạm ngữ nghĩa",
    "KNOWLEDGE_REPRESENTATION": "biểu diễn tri thức",
    "QUESTION_ANSWERING": "hỏi đáp",
    "LOCAL_DISCOURSE_CONTEXT": "tham chiếu",
    "REFERENCE": "tham chiếu",
    "ANAPHORA": "đồng tham chiếu",
    "CENTERING": "tham chiếu",
    "NOUN_PHRASES": "cụm danh từ",
    "VERB_PHRASES": "cụm động từ",
    "ADJECTIVE_PHRASES": "cụm tính từ",
    "ADVERBIAL_PHRASES": "cụm trạng từ",
    "STATISTICAL_NLP": "statistical NLP",
    "PYTHON_NLP": "Python NLP",
    "MACHINE_TRANSLATION": "machine translation",
    "AI_USAGE": "quy định về sử dụng AI",
    "LLM_RAG": "LLM/RAG",
    "MIDTERM": "midterm",
    "FINAL": "final",
    "LO1.1": "L.O.1.1",
    "LO2.1": "L.O.2.1",
    "LO2.2": "L.O.2.2",
    "LO2.3": "L.O.2.3",
    "LO2.4": "L.O.2.4",
    "LO2.5": "L.O.2.5",
    "LO3.1": "L.O.3.1",
    "LO3.2": "L.O.3.2",
    "LO3.3": "L.O.3.3",
    "LO3.4": "L.O.3.4",
    "LO4.1": "L.O.4.1",
    "LO4.2": "L.O.4.2",
}


def entity_category(entity_id):
    if not entity_id:
        return None
    if re.match(r"^CH\d{2}$", entity_id):
        return "chapter"
    if re.match(r"^WEEK_\d{2}$", entity_id):
        return "week"
    if entity_id in ("PART_I", "PART_II", "PART_III", "PART_IV"):
        return "part"
    if entity_id in ("BTL01",):
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


class DialogueState:
    def __init__(
        self,
        last_question=None,
        last_intent=None,
        last_entities=None,
        last_answer_entities=None,
        last_sources=None,
        ambiguous_candidates=None,
    ):
        self.last_question = last_question
        self.last_intent = last_intent
        self.last_entities = dict(last_entities or {})
        self.last_answer_entities = list(last_answer_entities or [])
        self.last_sources = list(last_sources or [])
        self._ambiguous_candidates = dict(ambiguous_candidates or {})

    def reset(self):
        """Clear all stored discourse context."""
        self.last_question = None
        self.last_intent = None
        self.last_entities = {}
        self.last_answer_entities = []
        self.last_sources = []
        self._ambiguous_candidates = {}

    def is_empty(self):
        return (
            not self.last_question
            and not self.last_intent
            and not self.last_entities
            and not self.last_answer_entities
        )

    def to_dict(self):
        """Return the exact schema required by the team plan."""
        return {
            "last_question": self.last_question,
            "last_intent": self.last_intent,
            "last_entities": deepcopy(self.last_entities),
            "last_answer_entities": list(self.last_answer_entities),
            "last_sources": list(self.last_sources),
        }

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data.copy()
        return cls(
            last_question=data.get("last_question"),
            last_intent=data.get("last_intent"),
            last_entities=data.get("last_entities"),
            last_answer_entities=data.get("last_answer_entities"),
            last_sources=data.get("last_sources"),
            ambiguous_candidates=data.get("_ambiguous_candidates"),
        )

    def copy(self):
        return DialogueState(
            last_question=self.last_question,
            last_intent=self.last_intent,
            last_entities=deepcopy(self.last_entities),
            last_answer_entities=list(self.last_answer_entities),
            last_sources=list(self.last_sources),
            ambiguous_candidates=deepcopy(self._ambiguous_candidates),
        )

    def has_ambiguous(self, category=None):
        if not self._ambiguous_candidates:
            return False
        if category and category in self._ambiguous_candidates:
            return len(self._ambiguous_candidates[category]) > 1
        return any(len(vals) > 1 for vals in self._ambiguous_candidates.values())

    def get_ambiguous_options(self, category=None):
        if category and category in self._ambiguous_candidates:
            return list(self._ambiguous_candidates[category])
        for vals in self._ambiguous_candidates.values():
            if len(vals) > 1:
                return list(vals)
        return []

    def update(self, question, intent, entities, answer_entities, sources, ambiguous_candidates=None):
        """Update state after an answered turn."""
        self.last_question = question
        if intent and intent != "UNKNOWN":
            self.last_intent = intent
        if entities:
            self.last_entities = dict(entities)
        self.last_answer_entities = list(dict.fromkeys(answer_entities or []))
        self.last_sources = list(dict.fromkeys(sources or []))
        self._ambiguous_candidates = dict(ambiguous_candidates or {})


class DialogueManager:
    """Resolve predetermined references in local dialogue before retrieval."""

    NEXT_CHAPTER_PATTERNS = [
        r"\bcòn\s+chương\s+sau\b\s*\??",
        r"\bthế\s+còn\s+chương\s+sau\b\s*\??",
        r"\bvậy\s+còn\s+chương\s+sau\b\s*\??",
        r"\bchương\s+sau\b",
        r"\bchương\s+tiếp\s+theo\b",
        r"\bchương\s+kế\s+tiếp\b",
        r"\bwhat\s+about\s+the\s+next\s+chapter\b\s*\??",
        r"\bnext\s+chapter\b",
    ]
    RESOURCE_PART_PATTERNS = [
        r"\btài\s+liệu\s+(?:tham\s+khảo\s+)?của\s+phần\s+này\b",
        r"\btài\s+liệu\s+(?:tham\s+khảo\s+)?cho\s+phần\s+này\b",
        r"\btài\s+liệu\s+phần\s+này\b",
        r"\btài\s+liệu\s+của\s+chương\s+này\b",
        r"\btài\s+liệu\s+của\s+nội\s+dung\s+này\b",
    ]
    CHAPTER_REF_PATTERNS = [
        r"\bchương\s+này\b",
        r"\bchương\s+đó\b",
        r"\bthis\s+chapter\b",
    ]
    WEEK_REF_PATTERNS = [
        r"\btuần\s+này\b",
        r"\btuần\s+đó\b",
        r"\bthis\s+week\b",
    ]
    PART_REF_PATTERNS = [
        r"\bphần\s+này\b",
        r"\bphần\s+đó\b",
        r"\bthis\s+part\b",
    ]
    CONTENT_REF_PATTERNS = [
        r"\bnội\s+dung\s+này\b",
        r"\bnội\s+dung\s+đó\b",
        r"\bchủ\s+đề\s+này\b",
        r"\bthis\s+content\b",
        r"\bthis\s+topic\b",
    ]
    PRONOUN_IT_PATTERNS = [
        r"(?:^|(?<=\s))nó(?=\s|[?.,!]|$)",
        r"\bcái\s+này\b",
        r"\bcái\s+đó\b",
        r"\bcái\s+kia\b",
        r"(?:^|(?<=\s))it(?=\s|[?.,!]|$)",
    ]

    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base
        self.kb_chapters = self._extract_kb_chapters(knowledge_base)

    @staticmethod
    def _extract_kb_chapters(knowledge_base):
        chapters = set()
        docs = getattr(knowledge_base, "documents", None) if knowledge_base else None
        if docs:
            for doc in docs:
                for eid in doc.get("entity_ids", []):
                    if re.match(r"^CH\d{2}$", str(eid)):
                        chapters.add(str(eid))
        if not chapters:
            chapters = {"CH{:02d}".format(i) for i in range(1, 13)}
        return chapters

    @staticmethod
    def detect_multiple_entities_in_question(question):
        """Detect if a question explicitly mentions multiple chapters/weeks/parts at once."""
        ambiguous = {}
        chapters = [
            "CH{:02d}".format(int(m))
            for m in re.findall(r"\b(?:chương|chapter)\s+0?(\d{1,2})\b", question or "", flags=re.IGNORECASE)
        ]
        unique_chapters = list(dict.fromkeys(chapters))
        if len(unique_chapters) > 1:
            ambiguous["chapter"] = unique_chapters

        weeks = [
            "WEEK_{:02d}".format(int(m))
            for m in re.findall(r"\b(?:tuần|week)\s+(?:thứ\s+)?0?(\d{1,2})\b", question or "", flags=re.IGNORECASE)
        ]
        unique_weeks = list(dict.fromkeys(weeks))
        if len(unique_weeks) > 1:
            ambiguous["week"] = unique_weeks

        parts = []
        for match in re.findall(r"\bphần\s+(I{1,3}|IV|[1-4])\b", question or "", flags=re.IGNORECASE):
            token = match.upper()
            mapping = {"1": "PART_I", "I": "PART_I", "2": "PART_II", "II": "PART_II", "3": "PART_III", "III": "PART_III", "4": "PART_IV", "IV": "PART_IV"}
            if token in mapping:
                parts.append(mapping[token])
        unique_parts = list(dict.fromkeys(parts))
        if len(unique_parts) > 1:
            ambiguous["part"] = unique_parts

        return ambiguous

    @staticmethod
    def _match_any(text, patterns):
        for pat in patterns:
            match = re.search(pat, text, flags=re.IGNORECASE)
            if match:
                return match
        return None

    def _infer_intent_from_tail(self, text, fallback_intent=None):
        norm = normalize(text)
        if any(kw in norm for kw in ("tai lieu", "giao trinh", "sach", "nguon hoc tap")):
            return "GET_RESOURCE"
        if any(kw in norm for kw in ("tuan nao", "tuan may", "lich hoc", "hoc gi", "hoc vao tuan")):
            return "GET_SCHEDULE"
        if any(kw in norm for kw in ("deadline", "han nop", "khi nao nop", "ngay nao")):
            return "GET_DEADLINE"
        if any(kw in norm for kw in ("xuat file", "nop file", "dau ra", "bai tap lon", "btl", "can lam gi", "nhiem vu")):
            return "GET_ASSIGNMENT"
        if any(kw in norm for kw in ("quy dinh", "duoc dung", "su dung ai")):
            return "GET_RULE"
        if any(kw in norm for kw in ("tin chi", "muc tieu", "midterm", "final", "bao nhieu phut")):
            return "GET_COURSE_INFO"
        if any(kw in norm for kw in ("noi ve", "chu de", "noi dung", "lien quan", "kien thuc", "phuong phap", "chuong nao")):
            return "GET_TOPIC"
        return fallback_intent or "GET_TOPIC"

    def resolve(self, question, state):
        """Resolve anaphora/reference expressions in `question` using `state`.

        Returns a dict:
        {
            "original_query": str,
            "resolved_query": str,
            "has_reference": bool,
            "reference_type": Optional[str],
            "reference_expression": Optional[str],
            "resolved_entities": dict,
            "resolved_intent": Optional[str],
            "status": "OK" | "NEED_CLARIFICATION" | "NOT_IN_KB",
            "clarification_message": Optional[str],
        }
        """
        q_clean = (question or "").strip()
        state = state if isinstance(state, DialogueState) else DialogueState.from_dict(state)

        # 1. Check "còn chương sau?" / "chương tiếp theo"
        match_next = self._match_any(q_clean, self.NEXT_CHAPTER_PATTERNS)
        if match_next:
            if state.is_empty():
                return self._clarify(
                    q_clean,
                    "NEXT_CHAPTER",
                    match_next.group(0),
                    "Bạn đang hỏi 'chương sau' nhưng chưa có chương nào trong ngữ cảnh trước đó. Bạn muốn hỏi tiếp theo của chương mấy?",
                )
            if state.has_ambiguous("chapter"):
                opts = ", ".join(ENTITY_DISPLAY.get(c, c) for c in state.get_ambiguous_options("chapter"))
                return self._clarify(
                    q_clean,
                    "NEXT_CHAPTER",
                    match_next.group(0),
                    "Lượt trước đề cập nhiều chương ({}). Bạn muốn hỏi chương tiếp theo của chương nào?".format(opts),
                )
            last_chapter = self._find_chapter_in_state(state)
            if not last_chapter:
                return self._clarify(
                    q_clean,
                    "NEXT_CHAPTER",
                    match_next.group(0),
                    "Chưa xác định được chương hiện tại trong ngữ cảnh để suy ra 'chương sau'. Bạn vui lòng nêu rõ số chương.",
                )
            chap_num = int(last_chapter.replace("CH", ""))
            next_num = chap_num + 1
            next_chapter_id = "CH{:02d}".format(next_num)
            next_chapter_label = "Chương {}".format(next_num)

            if next_chapter_id not in self.kb_chapters:
                max_chap = max((int(c.replace("CH", "")) for c in self.kb_chapters), default=12)
                return {
                    "original_query": q_clean,
                    "resolved_query": "{} nói về gì?".format(next_chapter_label),
                    "has_reference": True,
                    "reference_type": "NEXT_CHAPTER",
                    "reference_expression": match_next.group(0),
                    "resolved_entities": {"chapter": next_chapter_id},
                    "resolved_intent": state.last_intent or "GET_TOPIC",
                    "status": "NOT_IN_KB",
                    "clarification_message": (
                        "Sau {} là {}, tuy nhiên cơ sở tri thức môn học hiện chỉ có đến Chương {} "
                        "nên không có dữ liệu cho {}.".format(
                            ENTITY_DISPLAY.get(last_chapter, last_chapter),
                            next_chapter_label,
                            max_chap,
                            next_chapter_label,
                        )
                    ),
                }

            norm_q = normalize(q_clean)
            if "tuan" in norm_q or state.last_intent == "GET_SCHEDULE":
                resolved_query = "{} học vào tuần nào?".format(next_chapter_label)
                resolved_intent = "GET_SCHEDULE"
            elif "tai lieu" in norm_q or state.last_intent == "GET_RESOURCE":
                resolved_query = "Tài liệu tham khảo cho {} là gì?".format(next_chapter_label)
                resolved_intent = "GET_RESOURCE"
            else:
                resolved_query = "{} nói về gì?".format(next_chapter_label)
                resolved_intent = "GET_TOPIC"

            return {
                "original_query": q_clean,
                "resolved_query": resolved_query,
                "has_reference": True,
                "reference_type": "NEXT_CHAPTER",
                "reference_expression": match_next.group(0),
                "resolved_entities": {"chapter": next_chapter_id},
                "resolved_intent": resolved_intent,
                "status": "OK",
                "clarification_message": None,
            }

        # 2. Check "tài liệu của phần này" / "tài liệu cho phần này"
        match_res_part = self._match_any(q_clean, self.RESOURCE_PART_PATTERNS)
        if match_res_part:
            if state.is_empty():
                return self._clarify(
                    q_clean,
                    "RESOURCE_OF_THIS_PART",
                    match_res_part.group(0),
                    "Bạn muốn tìm tài liệu tham khảo của phần/chương nào? Vui lòng nêu rõ chủ đề hoặc chương học.",
                )
            if state.has_ambiguous():
                opts = ", ".join(ENTITY_DISPLAY.get(c, c) for c in state.get_ambiguous_options())
                return self._clarify(
                    q_clean,
                    "RESOURCE_OF_THIS_PART",
                    match_res_part.group(0),
                    "Ngữ cảnh trước có nhiều thực thể ({}). Bạn muốn hỏi tài liệu của phần nào?".format(opts),
                )
            target_key, target_val = self._pick_antecedent(
                state, preferred_keys=("chapter", "part", "topic", "week", "course")
            )
            if not target_val:
                return self._clarify(
                    q_clean,
                    "RESOURCE_OF_THIS_PART",
                    match_res_part.group(0),
                    "Không tìm thấy chủ đề hoặc chương trước đó để tra cứu tài liệu. Bạn vui lòng nêu rõ tên phần học.",
                )
            label = ENTITY_DISPLAY.get(target_val, target_val)
            resolved_entities = {target_key: target_val}

            if target_key == "topic" and target_val in ("STATISTICAL_NLP", "PYTHON_NLP", "MACHINE_TRANSLATION"):
                resolved_query = "Tài liệu tham khảo về {} là gì?".format(label)
            elif target_key == "topic":
                resolved_query = "Tài liệu nào nói về {}?".format(label)
            elif target_key == "week":
                related = [
                    ENTITY_DISPLAY.get(eid, eid)
                    for eid in state.last_answer_entities
                    if entity_category(str(eid)) in ("chapter", "topic")
                ][:2]
                for eid in state.last_answer_entities:
                    cat = entity_category(str(eid))
                    if cat in ("chapter", "topic") and cat not in resolved_entities:
                        resolved_entities[cat] = str(eid)
                if related:
                    resolved_query = "Tài liệu tham khảo cho {} ({}) là gì?".format(label, ", ".join(related))
                else:
                    resolved_query = "Tài liệu tham khảo cho {} là gì?".format(label)
            else:
                resolved_query = "Tài liệu tham khảo cho {} là gì?".format(label)

            return {
                "original_query": q_clean,
                "resolved_query": resolved_query,
                "has_reference": True,
                "reference_type": "RESOURCE_OF_THIS_PART",
                "reference_expression": match_res_part.group(0),
                "resolved_entities": resolved_entities,
                "resolved_intent": "GET_RESOURCE",
                "status": "OK",
                "clarification_message": None,
            }

        # 3. Check "chương này"
        match_chap = self._match_any(q_clean, self.CHAPTER_REF_PATTERNS)
        if match_chap:
            if state.is_empty():
                return self._clarify(
                    q_clean,
                    "THIS_CHAPTER",
                    match_chap.group(0),
                    "Bạn đang nói tới 'chương này' nhưng chưa đề cập chương nào trước đó. Bạn muốn hỏi chương mấy?",
                )
            if state.has_ambiguous("chapter"):
                opts = ", ".join(ENTITY_DISPLAY.get(c, c) for c in state.get_ambiguous_options("chapter"))
                return self._clarify(
                    q_clean,
                    "THIS_CHAPTER",
                    match_chap.group(0),
                    "Lượt trước nhắc tới nhiều chương ({}). Bạn muốn hỏi chương nào?".format(opts),
                )
            chap_id = self._find_chapter_in_state(state)
            if not chap_id:
                return self._clarify(
                    q_clean,
                    "THIS_CHAPTER",
                    match_chap.group(0),
                    "Trong ngữ cảnh vừa rồi chưa có thông tin chương cụ thể. Bạn muốn hỏi chương mấy?",
                )
            label = ENTITY_DISPLAY.get(chap_id, chap_id)
            resolved_query = self._substitute_and_canonicalize(
                q_clean, match_chap, label, entity_key="chapter", entity_id=chap_id, fallback_intent=state.last_intent
            )
            return {
                "original_query": q_clean,
                "resolved_query": resolved_query,
                "has_reference": True,
                "reference_type": "THIS_CHAPTER",
                "reference_expression": match_chap.group(0),
                "resolved_entities": {"chapter": chap_id},
                "resolved_intent": self._infer_intent_from_tail(resolved_query, state.last_intent),
                "status": "OK",
                "clarification_message": None,
            }

        # 4. Check "tuần này"
        match_week = self._match_any(q_clean, self.WEEK_REF_PATTERNS)
        if match_week:
            if state.is_empty():
                return self._clarify(
                    q_clean,
                    "THIS_WEEK",
                    match_week.group(0),
                    "Bạn đang nói tới 'tuần này' nhưng chưa đề cập tuần học nào trước đó. Bạn muốn hỏi tuần mấy?",
                )
            if state.has_ambiguous("week"):
                opts = ", ".join(ENTITY_DISPLAY.get(w, w) for w in state.get_ambiguous_options("week"))
                return self._clarify(
                    q_clean,
                    "THIS_WEEK",
                    match_week.group(0),
                    "Lượt trước nhắc tới nhiều tuần ({}). Bạn muốn hỏi tuần nào?".format(opts),
                )
            week_id = self._find_week_in_state(state)
            if not week_id:
                return self._clarify(
                    q_clean,
                    "THIS_WEEK",
                    match_week.group(0),
                    "Trong ngữ cảnh vừa rồi chưa có thông tin tuần học cụ thể. Bạn muốn hỏi tuần mấy?",
                )
            label = ENTITY_DISPLAY.get(week_id, week_id)
            resolved_query = self._substitute_and_canonicalize(
                q_clean, match_week, label, entity_key="week", entity_id=week_id, fallback_intent=state.last_intent
            )
            return {
                "original_query": q_clean,
                "resolved_query": resolved_query,
                "has_reference": True,
                "reference_type": "THIS_WEEK",
                "reference_expression": match_week.group(0),
                "resolved_entities": {"week": week_id},
                "resolved_intent": self._infer_intent_from_tail(resolved_query, state.last_intent or "GET_SCHEDULE"),
                "status": "OK",
                "clarification_message": None,
            }

        # 5. Check "phần này", "nội dung này", or "nó"
        match_part = self._match_any(q_clean, self.PART_REF_PATTERNS)
        match_content = self._match_any(q_clean, self.CONTENT_REF_PATTERNS)
        match_it = self._match_any(q_clean, self.PRONOUN_IT_PATTERNS)
        ref_match = match_part or match_content or match_it
        if ref_match:
            ref_type = "THIS_PART" if match_part else ("THIS_CONTENT" if match_content else "PRONOUN_IT")
            if state.is_empty():
                return self._clarify(
                    q_clean,
                    ref_type,
                    ref_match.group(0),
                    "Câu hỏi có từ tham chiếu '{}' nhưng chưa có ngữ cảnh trước đó. Bạn vui lòng nêu rõ chủ đề, phần hoặc chương muốn hỏi.".format(
                        ref_match.group(0)
                    ),
                )
            if state.has_ambiguous():
                opts = ", ".join(ENTITY_DISPLAY.get(c, c) for c in state.get_ambiguous_options())
                return self._clarify(
                    q_clean,
                    ref_type,
                    ref_match.group(0),
                    "Ngữ cảnh trước có nhiều khả năng tham chiếu ({}). Bạn muốn hỏi về thực thể nào?".format(opts),
                )

            norm_q = normalize(q_clean)
            if any(kw in norm_q for kw in ("xuat file", "nop file", "dau ra", "nhiem vu", "bai tap lon")):
                preferred = ("part", "assignment", "topic", "chapter", "week")
            elif any(kw in norm_q for kw in ("chuong nao", "o chuong nao", "thuoc chuong nao")):
                preferred = ("topic", "part", "week", "chapter")
            elif any(kw in norm_q for kw in ("tuan nao", "tuan may", "hoc vao tuan")):
                preferred = ("chapter", "topic", "part", "week")
            elif match_part:
                preferred = ("part", "chapter", "topic", "week", "assignment")
            elif match_content:
                preferred = ("topic", "chapter", "part", "week")
            else:
                preferred = ("chapter", "topic", "part", "week", "assignment", "course")

            target_key, target_val = self._pick_antecedent(state, preferred_keys=preferred)
            if not target_val:
                return self._clarify(
                    q_clean,
                    ref_type,
                    ref_match.group(0),
                    "Không tìm thấy thực thể phù hợp trong ngữ cảnh trước để giải nghĩa cho '{}'. Bạn vui lòng hỏi rõ hơn.".format(
                        ref_match.group(0)
                    ),
                )

            label = ENTITY_DISPLAY.get(target_val, target_val)
            resolved_query = self._substitute_and_canonicalize(
                q_clean, ref_match, label, entity_key=target_key, entity_id=target_val, fallback_intent=state.last_intent
            )
            return {
                "original_query": q_clean,
                "resolved_query": resolved_query,
                "has_reference": True,
                "reference_type": ref_type,
                "reference_expression": ref_match.group(0),
                "resolved_entities": {target_key: target_val},
                "resolved_intent": self._infer_intent_from_tail(resolved_query, state.last_intent),
                "status": "OK",
                "clarification_message": None,
            }

        # No reference detected
        return {
            "original_query": q_clean,
            "resolved_query": q_clean,
            "has_reference": False,
            "reference_type": None,
            "reference_expression": None,
            "resolved_entities": {},
            "resolved_intent": None,
            "status": "OK",
            "clarification_message": None,
        }

    def _substitute_and_canonicalize(self, question, match, label, entity_key, entity_id, fallback_intent=None):
        """Substitute the matched reference with `label` and normalize short conversational forms into grammar-friendly queries."""
        replaced = (question[:match.start()] + label + question[match.end():]).strip()
        # Strip leading conversational connectors like "Vậy ", "Thế ", "Còn " so CourseParser can parse the sentence
        replaced_clean = re.sub(r"^(?:vậy\s+thì|vậy\s+còn|thế\s+còn|vậy|thế|còn)\s+", "", replaced, flags=re.IGNORECASE).strip()
        norm = normalize(replaced_clean)

        if entity_key == "chapter":
            if norm in (
                normalize("{} nói về gì".format(label)),
                normalize("{} học gì".format(label)),
                normalize("nội dung {} là gì".format(label)),
                normalize("nội dung của {} là gì".format(label)),
            ):
                return "{} nói về gì?".format(label)
            if "tuan nao" in norm or "tuan may" in norm:
                return "{} học vào tuần nào?".format(label)
            if "tai lieu" in norm:
                return "Tài liệu tham khảo cho {} là gì?".format(label)

        if entity_key == "part":
            if "xuat file" in norm:
                return "{} cần xuất file gì?".format(label)
            if "nop" in norm and "file" in norm:
                return "{} cần nộp những file nào?".format(label)
            if "lam gi" in norm or "noi ve gi" in norm:
                return "{} cần làm gì?".format(label)

        if entity_key == "topic":
            if "chuong nao" in norm:
                return "{} được học ở chương nào?".format(label)
            if "tuan nao" in norm:
                return "Phần {} nằm ở tuần nào?".format(label)
            if "tai lieu" in norm:
                return "Tài liệu nào nói về {}?".format(label)

        if entity_key == "week":
            if "hoc gi" in norm or "noi ve gi" in norm:
                return "{} học gì?".format(label)

        return replaced_clean

    @staticmethod
    def _clarify(question, ref_type, ref_expr, message):
        return {
            "original_query": question,
            "resolved_query": question,
            "has_reference": True,
            "reference_type": ref_type,
            "reference_expression": ref_expr,
            "resolved_entities": {},
            "resolved_intent": None,
            "status": "NEED_CLARIFICATION",
            "clarification_message": message,
        }

    @staticmethod
    def _find_chapter_in_state(state):
        if "chapter" in state.last_entities:
            return state.last_entities["chapter"]
        for ent in state.last_answer_entities:
            if re.match(r"^CH\d{2}$", str(ent)):
                return str(ent)
        return None

    @staticmethod
    def _find_week_in_state(state):
        if "week" in state.last_entities:
            return state.last_entities["week"]
        for ent in state.last_answer_entities:
            if re.match(r"^WEEK_\d{2}$", str(ent)):
                return str(ent)
        return None

    @staticmethod
    def _pick_antecedent(state, preferred_keys):
        for key in preferred_keys:
            if key in state.last_entities and state.last_entities[key]:
                return key, state.last_entities[key]
        # Search in last_answer_entities by preferred category order
        for key in preferred_keys:
            for ent in state.last_answer_entities:
                if entity_category(str(ent)) == key:
                    return key, str(ent)
        for key, val in state.last_entities.items():
            if val:
                return key, val
        if state.last_answer_entities:
            first_ent = str(state.last_answer_entities[0])
            return entity_category(first_ent) or "topic", first_ent
        return None, None
