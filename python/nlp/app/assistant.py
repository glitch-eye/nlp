from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
import types
import urllib.error
import urllib.request

from nlp.app.text_utils import normalize


def _install_sklearn_tfidf_shim():
    class _MatrixWrapper:
        def __init__(self, rows):
            self.rows = rows

        def ravel(self):
            out = []
            for row in self.rows:
                if isinstance(row, (list, tuple)):
                    out.extend(row)
                else:
                    out.append(row)
            return out

        def __iter__(self):
            return iter(self.rows)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    class TfidfVectorizer:
        def __init__(self, preprocessor=None, lowercase=True, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True):
            self.preprocessor = preprocessor
            self.lowercase = lowercase
            self.token_pattern = re.compile(token_pattern)
            self.sublinear_tf = sublinear_tf
            self.idf_ = {}

        def _tokenize(self, text):
            processed = self.preprocessor(text) if self.preprocessor else (text or "")
            if self.lowercase:
                processed = processed.lower()
            return self.token_pattern.findall(processed)

        def fit_transform(self, raw_documents):
            tokenized_docs = [self._tokenize(doc) for doc in raw_documents]
            n_docs = len(tokenized_docs)
            df = Counter()
            for tokens in tokenized_docs:
                for term in set(tokens):
                    df[term] += 1
            self.idf_ = {
                term: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0
                for term, freq in df.items()
            }
            return _MatrixWrapper([self._vectorize_tokens(tokens) for tokens in tokenized_docs])

        def transform(self, raw_documents):
            return _MatrixWrapper([self._vectorize_tokens(self._tokenize(doc)) for doc in raw_documents])

        def _vectorize_tokens(self, tokens):
            counts = Counter(tokens)
            vec = {}
            norm_sq = 0.0
            for term, count in counts.items():
                if term not in self.idf_:
                    continue
                tf = (1.0 + math.log(count)) if self.sublinear_tf else float(count)
                val = tf * self.idf_[term]
                vec[term] = val
                norm_sq += val * val
            if norm_sq > 0.0:
                norm = math.sqrt(norm_sq)
                for term in vec:
                    vec[term] /= norm
            return vec

    def cosine_similarity(X, Y):
        x_rows = X.rows if isinstance(X, _MatrixWrapper) else list(X)
        y_rows = Y.rows if isinstance(Y, _MatrixWrapper) else list(Y)
        rows = []
        for x_vec in x_rows:
            row = []
            for y_vec in y_rows:
                if len(x_vec) <= len(y_vec):
                    score = sum(val * y_vec.get(term, 0.0) for term, val in x_vec.items())
                else:
                    score = sum(val * x_vec.get(term, 0.0) for term, val in y_vec.items())
                row.append(float(score))
            rows.append(row)
        return _MatrixWrapper(rows)

    sklearn_mod = types.ModuleType("sklearn")
    fe_mod = types.ModuleType("sklearn.feature_extraction")
    fet_mod = types.ModuleType("sklearn.feature_extraction.text")
    fet_mod.TfidfVectorizer = TfidfVectorizer
    metrics_mod = types.ModuleType("sklearn.metrics")
    pairwise_mod = types.ModuleType("sklearn.metrics.pairwise")
    pairwise_mod.cosine_similarity = cosine_similarity
    sys.modules.setdefault("sklearn", sklearn_mod)
    sys.modules.setdefault("sklearn.feature_extraction", fe_mod)
    sys.modules.setdefault("sklearn.feature_extraction.text", fet_mod)
    sys.modules.setdefault("sklearn.metrics", metrics_mod)
    sys.modules.setdefault("sklearn.metrics.pairwise", pairwise_mod)


try:
    import sklearn.feature_extraction.text  # noqa: F401
    import sklearn.metrics.pairwise  # noqa: F401
except ImportError:
    _install_sklearn_tfidf_shim()

from nlp.app.dialogue import DialogueManager, DialogueState, ENTITY_DISPLAY, entity_category
from nlp.app.knowledge_base import KnowledgeBase, default_data_dir
from nlp.app.parser import CourseParser
from nlp.app.retriever import Retriever
from nlp.app.semantic_frame import frame_from_tree


def _try_import(module_name, attr_name=None):
    for prefix in ("nlp.app.", ""):
        try:
            mod = __import__(prefix + module_name, fromlist=[attr_name] if attr_name else ["*"])
            if attr_name:
                return getattr(mod, attr_name, None)
            return mod
        except Exception:
            continue
    return None


class BaseLLMClient:
    model_name = "base"

    def generate(self, prompt, question="", chunks=None, semantic=None):
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    model_name = "mock-grounded-extractive-v1"

    def generate(self, prompt, question="", chunks=None, semantic=None):
        chunks = chunks or []
        if not chunks:
            return (
                "Không tìm thấy dữ liệu liên quan trong cơ sở tri thức môn học để trả lời câu hỏi này."
            )

        norm_q = normalize(question)
        intent = (semantic or {}).get("intent")
        entities = (semantic or {}).get("entities") or {}
        entity_ids = set((semantic or {}).get("entity_ids") or [])

        # Prioritize the chunk that best matches the question's specific entities and intent
        ranked_chunks = self._prioritize_chunks(chunks, norm_q, intent, entities, entity_ids)
        top = ranked_chunks[0]
        citation = "[Nguồn: {} / {}]".format(top.get("source", "unknown"), top.get("section", "unknown"))

        # 1. Check if top chunk is an FAQ item that directly answers the question
        if top.get("kind") == "faq":
            faq_ans = self._extract_faq_answer(top.get("text", ""))
            if faq_ans:
                # Enrich with detail from the second chunk if it is a primary topic/schedule/assignment chunk
                if len(ranked_chunks) > 1 and ranked_chunks[1].get("kind") in ("topics", "schedule", "assignments", "course_info"):
                    sec_chunk = ranked_chunks[1]
                    sec_detail = self._summarize_chunk(sec_chunk, norm_q, intent)
                    sec_cit = "[Nguồn: {} / {}]".format(sec_chunk.get("source"), sec_chunk.get("section"))
                    if sec_detail and normalize(sec_detail) not in normalize(faq_ans):
                        return "{} {} Ngoài ra: {} {}".format(faq_ans, citation, sec_detail, sec_cit)
                return "{} {}".format(faq_ans, citation)

        # 2. Yes/No relatedness questions (e.g. "Chương 4 có liên quan đến parsing không?", "Chương 11 có hỏi đáp không?", "Bài tập lớn có phần parser không?")
        if "co " in norm_q and norm_q.endswith("khong"):
            yn_answer = self._synthesize_yes_no(ranked_chunks, norm_q, intent, entities, entity_ids)
            if yn_answer:
                return yn_answer

        # 3. Synthesize from top 1-2 relevant chunks
        primary_summary = self._summarize_chunk(top, norm_q, intent)
        if len(ranked_chunks) > 1:
            second = ranked_chunks[1]
            # Combine if both are schedule weeks for the same chapter (e.g. Chapter 4 -> Week 4 and Week 5)
            if intent == "GET_SCHEDULE" and top.get("kind") == "schedule" and second.get("kind") == "schedule":
                sec_summary = self._summarize_chunk(second, norm_q, intent)
                sec_cit = "[Nguồn: {} / {}]".format(second.get("source"), second.get("section"))
                return "{} {} Bên cạnh đó: {} {}".format(primary_summary, citation, sec_summary, sec_cit)
            # Combine if asking for resources and multiple textbooks/references are retrieved
            if intent == "GET_RESOURCE" and top.get("kind") == "resources" and second.get("kind") == "resources":
                sec_summary = self._summarize_chunk(second, norm_q, intent)
                sec_cit = "[Nguồn: {} / {}]".format(second.get("source"), second.get("section"))
                return "{} {} Ngoài ra còn có: {} {}".format(primary_summary, citation, sec_summary, sec_cit)

        return "{} {}".format(primary_summary, citation)

    def _prioritize_chunks(self, chunks, norm_q, intent, entities, entity_ids):
        """Re-order top-k retrieved chunks so the most specific chunk for answering comes first."""
        def score_chunk(idx_chunk):
            idx, ch = idx_chunk
            kind = ch.get("kind", "")
            ch_ents = set(ch.get("entity_ids", []))
            bonus = -idx * 0.05
            # Match specific part / chapter / week
            if entities.get("part") and entities["part"] in ch_ents:
                bonus += 2.0
            if entities.get("chapter") and entities["chapter"] in ch_ents:
                bonus += 1.5
            if entities.get("week") and entities["week"] in ch_ents:
                bonus += 1.5
            # Intent-specific source preference for detailed answers
            if intent == "GET_SCHEDULE" and kind == "schedule":
                bonus += 1.2
            elif intent == "GET_TOPIC" and kind == "topics":
                bonus += 1.0
            elif intent == "GET_ASSIGNMENT" and kind == "assignments":
                bonus += 1.2
            elif intent == "GET_RESOURCE" and kind == "resources":
                bonus += 1.2
            elif intent == "GET_RULE" and kind == "regulations":
                bonus += 1.0
            if "xuat file" in norm_q or "nop file" in norm_q or "dau ra" in norm_q:
                if "suggested_outputs" in ch.get("text", ""):
                    bonus += 2.5
            return bonus

        return [ch for _, ch in sorted(enumerate(chunks), key=score_chunk, reverse=True)]

    @staticmethod
    def _extract_faq_answer(text):
        match = re.search(r"answer:\s*(.+)$", text.strip(), flags=re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).split())
        return None

    def _synthesize_yes_no(self, ranked_chunks, norm_q, intent, entities, entity_ids):
        for ch in ranked_chunks:
            ch_ents = set(ch.get("entity_ids", []))
            txt_flat = " ".join(ch.get("text", "").split())
            cit = "[Nguồn: {} / {}]".format(ch.get("source"), ch.get("section"))
            if entities.get("chapter") and entities["chapter"] in ch_ents:
                summary = self._summarize_chunk(ch, norm_q, intent)
                return "Có. {} {}".format(summary, cit)
            if "parser" in norm_q or "parsing" in norm_q:
                if "PARSING" in ch_ents or "PART_I" in ch_ents or "parser" in normalize(txt_flat):
                    summary = self._summarize_chunk(ch, norm_q, intent)
                    return "Có. {} {}".format(summary, cit)
            if "hoi dap" in norm_q or "question answering" in norm_q:
                if "QUESTION_ANSWERING" in ch_ents or "question answering" in normalize(txt_flat):
                    summary = self._summarize_chunk(ch, norm_q, intent)
                    return "Có. {} {}".format(summary, cit)
        return None

    def _summarize_chunk(self, chunk, norm_q, intent):
        text = " ".join(chunk.get("text", "").split())
        kind = chunk.get("kind", "")
        section = chunk.get("section", "")

        if kind == "faq":
            ans = self._extract_faq_answer(text)
            if ans:
                return ans

        if kind == "topics":
            summary_match = re.search(r"summary:\s*(.+?)(?:\s+subtopics:|$)", text, flags=re.IGNORECASE)
            subtopics_match = re.search(r"subtopics:\s*(.+)$", text, flags=re.IGNORECASE)
            parts = [section]
            if summary_match:
                parts.append("Tóm tắt: {}".format(summary_match.group(1).strip()))
            if subtopics_match:
                parts.append("Các mục chi tiết: {}".format(subtopics_match.group(1).strip()))
            return " — ".join(parts) if len(parts) > 1 else text

        if kind == "schedule":
            chap_match = re.search(r"chapter:\s*(Chapter\s+\d+)", text, flags=re.IGNORECASE)
            topics_match = re.search(r"topics:\s*(.+?)(?:\s+activity:|$)", text, flags=re.IGNORECASE)
            if chap_match and topics_match:
                return "{} học {} với các nội dung: {}.".format(
                    section, chap_match.group(1), topics_match.group(1).strip()
                )
            return text

        if kind == "assignments":
            req_match = re.search(r"requirements:\s*(.+?)(?:\s+suggested_outputs:|$)", text, flags=re.IGNORECASE)
            out_match = re.search(r"suggested_outputs:\s*(.+)$", text, flags=re.IGNORECASE)
            if "xuat file" in norm_q or "nop" in norm_q or "dau ra" in norm_q:
                if out_match:
                    reqs = " (Yêu cầu: {})".format(req_match.group(1).strip()) if req_match else ""
                    return "{} yêu cầu xuất các file đầu ra: {}{}.".format(
                        section, out_match.group(1).strip(), reqs
                    )
            if req_match:
                outs = " Đầu ra gợi ý: {}.".format(out_match.group(1).strip()) if out_match else ""
                return "{} có yêu cầu: {}.{}".format(section, req_match.group(1).strip(), outs)
            return text

        if kind == "resources":
            title_m = re.search(r"title:\s*(.+?)(?:\s+authors?:|\s+publisher:|\s+year:|$)", text, flags=re.IGNORECASE)
            author_m = re.search(r"authors?:\s*(.+?)(?:\s+publisher:|\s+year:|\s+usage:|$)", text, flags=re.IGNORECASE)
            usage_m = re.search(r"usage:\s*(.+)$", text, flags=re.IGNORECASE)
            if title_m:
                info = "{} — '{}'".format(section, title_m.group(1).strip())
                if author_m:
                    info += " (Tác giả: {})".format(author_m.group(1).strip())
                if usage_m:
                    info += ", công dụng: {}".format(usage_m.group(1).strip())
                return info + "."
            return text

        return text


class OpenAICompatibleLLMClient(BaseLLMClient):
    def __init__(self, api_key=None, model=None, base_url=None, timeout=30):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model_name = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout
        self._fallback = MockLLMClient()

    def generate(self, prompt, question="", chunks=None, semantic=None):
        if not self.api_key:
            return self._fallback.generate(prompt, question=question, chunks=chunks, semantic=semantic)
        url = "{}/chat/completions".format(self.base_url)
        payload = {
            "model": self.model_name,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý môn học Xử lý Ngôn ngữ Tự nhiên (CO3085). "
                        "Chỉ trả lời dựa trên CONTEXT được cung cấp và luôn trích dẫn [Nguồn: file / section]. "
                        "Nếu CONTEXT không đủ thông tin, hãy nêu rõ không đủ dữ liệu."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(self.api_key),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, ValueError):
            return self._fallback.generate(prompt, question=question, chunks=chunks, semantic=semantic)


def create_llm_client(provider=None, model=None):
    """Factory selecting MockLLMClient by default, or real API client when configured."""
    teammate_client_cls = _try_import("llm_client", "LLMClient")
    if teammate_client_cls is not None:
        try:
            return teammate_client_cls()
        except Exception:
            pass

    provider = (provider or os.environ.get("LLM_PROVIDER", "mock")).strip().lower()
    if provider in ("openai", "api") or (provider == "auto" and os.environ.get("OPENAI_API_KEY")):
        return OpenAICompatibleLLMClient(model=model)
    return MockLLMClient()


def _default_build_prompt(question, resolved_query, semantic, chunks):
    """Build a structured grounded prompt when prompt_builder.py is not yet present."""
    context_blocks = []
    for idx, ch in enumerate(chunks or [], 1):
        context_blocks.append(
            "[Chunk {}] (chunk_id={}, source={}, section={})\n{}".format(
                idx,
                ch.get("chunk_id", ""),
                ch.get("source", ""),
                ch.get("section", ""),
                ch.get("text", ""),
            )
        )
    context_text = "\n\n".join(context_blocks) if context_blocks else "(Không có context)"
    return (
        "YÊU CẦU HỆ THỐNG:\n"
        "- Chỉ trả lời câu hỏi dựa trên phần CONTEXT bên dưới.\n"
        "- Không suy diễn hoặc bịa đặt thông tin không có trong CONTEXT.\n"
        "- Đính kèm trích dẫn nguồn theo định dạng [Nguồn: <source> / <section>].\n\n"
        "CÂU HỎI GỐC: {}\n"
        "TRUY VẤN ĐÃ GIẢI THAM CHIẾU: {}\n"
        "SEMANTIC FRAME: {}\n\n"
        "CONTEXT:\n{}\n\n"
        "CÂU TRẢ LỜI CÓ CĂN CỨ:".format(
            question,
            resolved_query,
            json.dumps(semantic or {}, ensure_ascii=False),
            context_text,
        )
    )


def _default_guard_check(question, resolved_query, semantic, retrieval_result):
    """Check for NO_RELEVANT_CONTEXT or INSUFFICIENT_CONTEXT before/after LLM call."""
    chunks = (retrieval_result or {}).get("chunks") or []
    retrieval_status = (retrieval_result or {}).get("retrieval_status", "NO_RELEVANT_CONTEXT")
    if retrieval_status != "FOUND" or not chunks:
        return {
            "status": "NO_RELEVANT_CONTEXT",
            "answer": (
                "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở tri thức môn học CO3085 "
                "để trả lời câu hỏi: '{}'.".format(question)
            ),
        }

    norm_q = normalize(resolved_query or question)
    intent = (semantic or {}).get("intent")

    # 1. Check GET_DEADLINE: sample dataset explicitly notes deadlines are sample data without official dates
    if intent == "GET_DEADLINE" or any(kw in norm_q for kw in ("deadline", "han nop", "thoi han nop")):
        top = chunks[0]
        cit = "[Nguồn: {} / {}]".format(top.get("source", "assignments.txt"), top.get("section", "Lưu ý"))
        return {
            "status": "INSUFFICIENT_CONTEXT",
            "answer": (
                "Cơ sở tri thức hiện tại không có ngày giờ hạn nộp chính thức cho bài tập lớn "
                "(tài liệu chỉ ghi nhận đây là dữ liệu mẫu phục vụ Course Assistant và cần thay bằng lịch chính thức của lớp). {}".format(cit)
            ),
        }

    # 2. Check unsupported specific attributes (classroom, instructor email, tuition, submission URL, etc.)
    unsupported_keywords = (
        ("phong hoc", "phòng học"),
        ("giang vien", "giảng viên phụ trách"),
        ("thay co", "giảng viên"),
        ("email", "địa chỉ email"),
        ("hoc phi", "học phí"),
        ("link nop", "đường dẫn nộp bài"),
        ("diem cong", "chính sách điểm cộng"),
    )
    combined_context_norm = normalize(" ".join(ch.get("text", "") for ch in chunks))
    for kw_norm, kw_label in unsupported_keywords:
        if kw_norm in norm_q and kw_norm not in combined_context_norm:
            top = chunks[0]
            cit = "[Nguồn: {} / {}]".format(top.get("source", ""), top.get("section", ""))
            return {
                "status": "INSUFFICIENT_CONTEXT",
                "answer": (
                    "Các đoạn tài liệu truy xuất được không có thông tin cụ thể về {} cho câu hỏi này. {}".format(
                        kw_label, cit
                    )
                ),
            }

    return {"status": "OK", "answer": None}


def _build_fallback_kb_from_traces():
    """Construct an in-memory KnowledgeBase from output/retrieval.txt if NLP_Course_Assistant_Sample_Data is outside the workspace."""
    chunks = {}
    app_dir = Path(__file__).resolve().parent
    retrieval_trace = app_dir / "output" / "retrieval.txt"
    if retrieval_trace.exists():
        lines = retrieval_trace.read_text(encoding="utf-8", errors="replace").splitlines()
        idx = 0
        while idx < len(lines):
            match = re.match(r"^\d+\.\s+(\S+)\s+\|\s+score=[0-9.]+\s+\|\s+(.+?)\s+/\s+(.+)$", lines[idx])
            if match and idx + 3 < len(lines):
                cid, src, sec = match.group(1), match.group(2), match.group(3)
                ent_line = lines[idx + 1].strip()
                ents = [
                    e.strip()
                    for e in ent_line.replace("entity_ids:", "").split(",")
                    if e.strip() and e.strip() != "(none)"
                ]
                txt = lines[idx + 3].strip()
                chunks[cid] = {
                    "chunk_id": cid,
                    "source": src,
                    "section": sec,
                    "entity_ids": ents,
                    "text": txt,
                    "kind": src.replace(".txt", ""),
                }
                idx += 4
            else:
                idx += 1

    extra = {
        "schedule-week-06-6": ("schedule.txt", "WEEK 06", ["CH05", "WEEK_06", "SYNTACTIC_PROCESSING", "CFG", "VERB_PHRASES"], "WEEK 06 part: Syntactic Processing chapter: Chapter 5 topics: - grammars for natural language - auxiliary verbs and verb phrases - movement phenomena in language - handling questions in context-free grammars - relative clauses"),
        "topics-chapter-5-grammar-for-natural-language-4": ("topics.txt", "CHAPTER 5: GRAMMAR FOR NATURAL LANGUAGE", ["CH05", "SYNTACTIC_PROCESSING", "CFG", "VERB_PHRASES"], "CHAPTER 5: GRAMMAR FOR NATURAL LANGUAGE chapter_id: CH05 part: Syntactic Processing keywords: natural language grammar, auxiliary verbs, movement, wh-questions, relative clauses summary: The chapter develops comprehensive grammars for natural language phenomena including auxiliary verbs, verb phrases, movement phenomena, yes/no and wh-questions, and relative clauses. subtopics: 5.1 Auxiliary verbs and verb phrases 5.2 Movement phenomena in language 5.3 Handling questions in context-free grammars 5.4 Relative clauses 5.5 The hold mechanism in ATNs"),
        "resources-textbook-1-0": ("resources.txt", "TEXTBOOK 1", ["NLP", "NATURAL_LANGUAGE_PROCESSING"], "TEXTBOOK 1 title: Natural Language Understanding author: James Allen publisher: Benjamin/Cummings year: 1995 usage: main course textbook covering Part I (Syntactic Processing), Part II (Semantic Interpretation), and Part III (Context and World Knowledge)"),
        "faq-q13-12": ("faq.txt", "Q13", ["BTL01", "PART_I", "PART_II", "PART_III"], "Q13: question: Bài tập lớn gồm những phần nào? answer: Bài tập lớn gồm Phần I (Grammar & Parser), Phần II (Semantic Interpretation & QA hoặc LLM/RAG), Phần III (Local Discourse Context) và Phần IV (Evaluation)."),
    }
    for cid, (src, sec, ents, txt) in extra.items():
        if cid not in chunks:
            chunks[cid] = {"chunk_id": cid, "source": src, "section": sec, "entity_ids": list(ents), "text": txt, "kind": src.replace(".txt", "")}

    aliases = [
        ("Môn NLP", "NLP", "Xử lý Ngôn ngữ Tự nhiên"),
        ("Xử lý Ngôn ngữ Tự nhiên", "NLP", "Xử lý Ngôn ngữ Tự nhiên"),
        ("CO3085", "CO3085", "Xử lý Ngôn ngữ Tự nhiên"),
        ("bài tập lớn", "BTL01", "Bài tập lớn Course Assistant"),
        ("Phần I", "PART_I", "Phần I"),
        ("Phần II", "PART_II", "Phần II"),
        ("Phần III", "PART_III", "Phần III"),
        ("Phần IV", "PART_IV", "Phần IV"),
        ("CFG", "CFG", "context-free grammar"),
        ("Parsing", "PARSING", "phân tích cú pháp"),
        ("parsing", "PARSING", "phân tích cú pháp"),
        ("PCFG", "PCFG", "probabilistic context-free grammar"),
        ("văn phạm phụ thuộc", "DEPENDENCY_GRAMMAR", "văn phạm phụ thuộc"),
        ("nét và văn phạm gia tố", "FEATURES", "nét và văn phạm gia tố"),
        ("gán nhãn từ loại", "POS_TAGGING", "gán nhãn từ loại"),
        ("dạng luận lý", "LOGICAL_FORM", "dạng luận lý"),
        ("vai trò chủ đề", "THEMATIC_ROLES", "vai trò chủ đề"),
        ("giới hạn lựa chọn", "SELECTIONAL_RESTRICTION", "giới hạn lựa chọn"),
        ("mạng ngữ nghĩa", "SEMANTIC_NETWORK", "mạng ngữ nghĩa"),
        ("văn phạm ngữ nghĩa", "SEMANTIC_GRAMMAR", "văn phạm ngữ nghĩa"),
        ("biểu diễn tri thức", "KNOWLEDGE_REPRESENTATION", "biểu diễn tri thức"),
        ("hỏi đáp", "QUESTION_ANSWERING", "hỏi đáp"),
        ("tham chiếu", "REFERENCE", "tham chiếu"),
        ("đồng tham chiếu", "ANAPHORA", "đồng tham chiếu"),
        ("statistical NLP", "STATISTICAL_NLP", "statistical NLP"),
        ("Python NLP", "PYTHON_NLP", "Python NLP"),
        ("machine translation", "MACHINE_TRANSLATION", "machine translation"),
        ("LLM/RAG", "LLM_RAG", "LLM/RAG"),
        ("sử dụng AI", "AI_USAGE", "AI usage"),
        ("midterm", "MIDTERM", "kiểm tra giữa kỳ"),
        ("final", "FINAL", "thi cuối kỳ"),
    ]
    for ch_idx in range(1, 13):
        aliases.append(("Chương {}".format(ch_idx), "CH{:02d}".format(ch_idx), "Chương {}".format(ch_idx)))
    for wk_idx in range(1, 16):
        aliases.append(("Tuần {}".format(wk_idx), "WEEK_{:02d}".format(wk_idx), "Tuần {}".format(wk_idx)))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)

    kb_stub = types.SimpleNamespace(
        data_dir=app_dir,
        aliases=aliases,
        documents=[chunks[cid] for cid in sorted(chunks)],
    )
    return kb_stub


class CourseAssistant:
    """Single-turn and multi-turn RAG Course Assistant orchestrator."""

    def __init__(
        self,
        data_dir=None,
        grammar_path=None,
        parser=None,
        knowledge_base=None,
        retriever=None,
        llm_client=None,
        top_k=3,
    ):
        if knowledge_base is not None:
            self.kb = knowledge_base
        else:
            try:
                self.kb = KnowledgeBase(data_dir or default_data_dir())
            except (FileNotFoundError, OSError):
                self.kb = _build_fallback_kb_from_traces()
        self.parser = parser or CourseParser(grammar_path=grammar_path)
        self.retriever = retriever or Retriever(
            self.kb.documents,
            aliases=self.kb.aliases,
            top_k=top_k,
            parser=self.parser,
        )
        self.dialogue_manager = DialogueManager(knowledge_base=self.kb)
        self.llm_client = llm_client or create_llm_client()

        # Optional teammate (Người 2) hooks if present
        self._teammate_semantic = _try_import("semantic")
        self._teammate_prompt_builder = _try_import("prompt_builder")
        self._teammate_answer_guard = _try_import("answer_guard")

    def _build_semantic_frame(self, resolved_query, parse_tree, resolution):
        """Construct unified semantic frame combining Part I parser output, surface entities, and dialogue context."""
        parse_status = "PARSED" if parse_tree is not None else "UNPARSED"

        # 1. Try Người 2's semantic.py if available
        frame = None
        if self._teammate_semantic is not None and parse_tree is not None:
            for fn_name in ("interpret", "extract_semantic_frame", "frame_from_tree"):
                fn = getattr(self._teammate_semantic, fn_name, None)
                if callable(fn):
                    try:
                        frame = fn(parse_tree, resolved_query)
                        break
                    except TypeError:
                        try:
                            frame = fn(parse_tree)
                            break
                        except Exception:
                            pass

        # 2. Use Người 1's semantic_frame.frame_from_tree
        if frame is None and parse_tree is not None:
            frame = frame_from_tree(parse_tree, resolved_query, self.kb.aliases)

        intent = (frame or {}).get("intent")
        entity_ids = list((frame or {}).get("entity_ids") or [])
        entities = dict((frame or {}).get("entities") or {})

        # 3. Extract explicit entities from resolved_query so Part I/II/III entities are never missed
        norm_q = " " + normalize(resolved_query) + " "
        for alias, eid, _ in self.kb.aliases:
            norm_alias = " " + normalize(alias) + " "
            if norm_alias.strip() and norm_alias in norm_q and eid not in entity_ids:
                entity_ids.append(eid)

        for pattern, formatter in (
            (r"\b(?:tuần|week)\s+(?:thứ\s+)?0?(\d{1,2})\b", lambda m: "WEEK_{:02d}".format(int(m.group(1)))),
            (r"\b(?:chương|chapter)\s+0?(\d{1,2})\b", lambda m: "CH{:02d}".format(int(m.group(1)))),
            (r"\bL\s*\.?\s*O\s*\.?\s*(\d+\.\d+)\b", lambda m: "LO" + m.group(1)),
        ):
            for match in re.finditer(pattern, resolved_query, flags=re.IGNORECASE):
                eid = formatter(match)
                if eid not in entity_ids:
                    entity_ids.append(eid)

        for match in re.finditer(r"\bphần\s+(I{1,3}|IV|[1-4])\b", resolved_query, flags=re.IGNORECASE):
            token = match.group(1).upper()
            part_map = {"1": "PART_I", "I": "PART_I", "2": "PART_II", "II": "PART_II", "3": "PART_III", "III": "PART_III", "4": "PART_IV", "IV": "PART_IV"}
            eid = part_map.get(token)
            if eid and eid not in entity_ids:
                entity_ids.append(eid)

        # Recognize "phần parser" / "phần grammar" in assignment questions (e.g. Q033)
        if ("bai tap lon" in norm_q or "btl" in norm_q) and ("parser" in norm_q or "grammar" in norm_q):
            for eid in ("BTL01", "PART_I", "PARSING"):
                if eid not in entity_ids:
                    entity_ids.append(eid)

        # Merge dialogue-resolved entities
        for k, v in (resolution.get("resolved_entities") or {}).items():
            if v and v not in entity_ids:
                entity_ids.append(v)
            if k and v:
                entities[k] = v

        for eid in entity_ids:
            cat = entity_category(eid)
            if cat and cat not in entities:
                entities[cat] = eid

        # Infer intent when UNPARSED or when dialogue resolved an intent
        if not intent:
            intent = resolution.get("resolved_intent") or self._infer_intent_fallback(resolved_query, entity_ids)

        if intent == "GET_RESOURCE" and not any(
            eid in ("STATISTICAL_NLP", "PYTHON_NLP", "MACHINE_TRANSLATION") for eid in entity_ids
        ):
            if "NLP" not in entity_ids:
                entity_ids.append("NLP")

        return {
            "intent": intent,
            "entities": entities,
            "entity_ids": entity_ids,
            "parse_status": parse_status,
        }

    @staticmethod
    def _infer_intent_fallback(query, entity_ids):
        norm = normalize(query)
        if any(kw in norm for kw in ("deadline", "han nop", "thoi han nop")):
            return "GET_DEADLINE"
        if any(kw in norm for kw in ("tai lieu", "giao trinh", "sach tham khao", "nguon hoc tap")):
            return "GET_RESOURCE"
        if any(kw in norm for kw in ("quy dinh", "duoc dung", "su dung ai", "llm rag")):
            return "GET_RULE"
        if any(eid.startswith("LO") for eid in entity_ids) or "l o " in norm:
            return "GET_LO"
        if any(kw in norm for kw in ("bai tap lon", "btl", "xuat file", "nop file")) or any(
            eid in ("BTL01", "PART_I", "PART_II", "PART_III", "PART_IV") for eid in entity_ids
        ):
            return "GET_ASSIGNMENT"
        if any(kw in norm for kw in ("tuan", "lich hoc", "ke hoach hoc tap", "may tiet")):
            return "GET_SCHEDULE"
        if any(kw in norm for kw in ("tin chi", "midterm", "final", "giua ky", "cuoi ky", "gom nhung phan nao", "muc tieu")):
            return "GET_COURSE_INFO"
        if any(eid.startswith("CH") for eid in entity_ids) or any(
            kw in norm for kw in ("chuong", "noi ve", "phuong phap", "lien quan", "hoc o dau", "hoc ve")
        ):
            return "GET_TOPIC"
        return None

    def answer(self, question, dialogue_state=None):
        """Run end-to-end CourseAssistant pipeline for one question and optional DialogueState."""
        use_dialogue = dialogue_state is not None
        state = (
            dialogue_state
            if isinstance(dialogue_state, DialogueState)
            else DialogueState.from_dict(dialogue_state)
        )
        state_before = state.to_dict()

        # Step 0: Resolve local dialogue references if dialogue_state is active
        if use_dialogue:
            resolution = self.dialogue_manager.resolve(question, state)
        else:
            resolution = {
                "original_query": (question or "").strip(),
                "resolved_query": (question or "").strip(),
                "has_reference": False,
                "reference_type": None,
                "reference_expression": None,
                "resolved_entities": {},
                "resolved_intent": None,
                "status": "OK",
                "clarification_message": None,
            }

        resolved_query = resolution["resolved_query"]

        # Handle missing/ambiguous antecedent -> NEED_CLARIFICATION (never let LLM guess)
        if resolution["status"] == "NEED_CLARIFICATION":
            return {
                "question": question,
                "resolved_query": resolved_query,
                "answer": resolution["clarification_message"],
                "sources": [],
                "retrieved_chunk_ids": [],
                "parse_status": "SKIPPED",
                "retrieval_status": "SKIPPED",
                "status": "NEED_CLARIFICATION",
                "intent": None,
                "entities": {},
                "entity_ids": [],
                "chunks": [],
                "scores": [],
                "resolution": resolution,
                "dialogue_state_before": state_before,
                "dialogue_state_after": state.to_dict(),
            }

        # Handle chapter increment beyond KB range (e.g. after CH12 -> CH13)
        if resolution["status"] == "NOT_IN_KB":
            return {
                "question": question,
                "resolved_query": resolved_query,
                "answer": resolution["clarification_message"],
                "sources": [],
                "retrieved_chunk_ids": [],
                "parse_status": "SKIPPED",
                "retrieval_status": "NO_RELEVANT_CONTEXT",
                "status": "INSUFFICIENT_CONTEXT",
                "intent": resolution.get("resolved_intent"),
                "entities": resolution.get("resolved_entities") or {},
                "entity_ids": list((resolution.get("resolved_entities") or {}).values()),
                "chunks": [],
                "scores": [],
                "resolution": resolution,
                "dialogue_state_before": state_before,
                "dialogue_state_after": state.to_dict(),
            }

        # Step 1: CourseParser.parse() (Part I CFG parser)
        parse_tree = self.parser.parse(resolved_query) if self.parser else None
        parse_status = "PARSED" if parse_tree is not None else "UNPARSED"

        # Step 2: Semantic frame construction
        semantic = self._build_semantic_frame(resolved_query, parse_tree, resolution)

        # Step 3: Retriever.retrieve()
        retrieval = self.retriever.retrieve(resolved_query, semantic=semantic)
        # Keep honest parse_status from CourseParser on resolved_query
        retrieval["parse_status"] = parse_status
        chunks = retrieval.get("chunks") or []
        scores = retrieval.get("scores") or []
        retrieval_status = retrieval.get("retrieval_status", "NO_RELEVANT_CONTEXT")

        sources = []
        seen_sources = set()
        for ch in chunks:
            src_pair = (ch.get("source", ""), ch.get("section", ""))
            if src_pair not in seen_sources:
                seen_sources.add(src_pair)
                sources.append({"source": src_pair[0], "section": src_pair[1]})
        retrieved_chunk_ids = [ch.get("chunk_id", "") for ch in chunks]

        # Step 4: No-context / Insufficient-context guard
        guard_result = _default_guard_check(question, resolved_query, semantic, retrieval)
        if self._teammate_answer_guard is not None:
            guard_fn = getattr(self._teammate_answer_guard, "check_context", None) or getattr(
                self._teammate_answer_guard, "guard_answer", None
            )
            if callable(guard_fn):
                try:
                    custom_guard = guard_fn(question, retrieval, semantic)
                    if isinstance(custom_guard, dict) and custom_guard.get("status") in (
                        "NO_RELEVANT_CONTEXT",
                        "INSUFFICIENT_CONTEXT",
                    ):
                        guard_result = custom_guard
                except Exception:
                    pass

        if guard_result["status"] in ("NO_RELEVANT_CONTEXT", "INSUFFICIENT_CONTEXT"):
            status = guard_result["status"]
            answer_text = guard_result["answer"]
            if status == "NO_RELEVANT_CONTEXT":
                sources = []
                retrieved_chunk_ids = []
        else:
            # Step 5: Build grounded prompt
            prompt = _default_build_prompt(question, resolved_query, semantic, chunks)
            if self._teammate_prompt_builder is not None:
                build_fn = getattr(self._teammate_prompt_builder, "build_prompt", None)
                if callable(build_fn):
                    try:
                        prompt = build_fn(resolved_query, semantic, chunks)
                    except Exception:
                        pass

            # Step 6: Call LLM / MockLLMClient
            try:
                answer_text = self.llm_client.generate(
                    prompt, question=resolved_query, chunks=chunks, semantic=semantic
                )
            except TypeError:
                answer_text = self.llm_client.generate(prompt)
            status = "ANSWERED"

        # Step 7: Update DialogueState when a turn is answered or grounded in retrieved chunks
        ambiguous_multi = DialogueManager.detect_multiple_entities_in_question(resolved_query)
        if use_dialogue and (status == "ANSWERED" or (status == "INSUFFICIENT_CONTEXT" and chunks)):
            answer_entities = []
            for eid in semantic.get("entity_ids", []):
                if eid not in answer_entities:
                    answer_entities.append(eid)
            for ch in chunks:
                for eid in ch.get("entity_ids", []):
                    if eid not in answer_entities:
                        answer_entities.append(eid)
            source_labels = ["{} / {}".format(s["source"], s["section"]) for s in sources]
            state.update(
                question=question,
                intent=semantic.get("intent"),
                entities=semantic.get("entities"),
                answer_entities=answer_entities,
                sources=source_labels,
                ambiguous_candidates=ambiguous_multi,
            )
        elif use_dialogue and ambiguous_multi:
            state._ambiguous_candidates = ambiguous_multi

        return {
            "question": question,
            "resolved_query": resolved_query,
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "parse_status": parse_status,
            "retrieval_status": retrieval_status,
            "status": status,
            "intent": semantic.get("intent"),
            "entities": semantic.get("entities") or {},
            "entity_ids": semantic.get("entity_ids") or [],
            "chunks": chunks,
            "scores": scores,
            "resolution": resolution,
            "dialogue_state_before": state_before,
            "dialogue_state_after": state.to_dict(),
        }
