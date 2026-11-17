"""Answer guard (Member 2).

Two hooks, both optional for ``assistant.py``:

* ``check_context(question, retrieval, semantic)``  -- BEFORE the LLM call.
  Catches questions the retriever scored above threshold but that are not
  supported by the retrieved text (out-of-domain paraphrases).
  Returns {"status": "OK"} when unsure, so existing guards keep working.
* ``check_answer(answer, chunks)``  -- AFTER the LLM call.
  Handles the INSUFFICIENT_CONTEXT sentinel, validates every
  ``[Nguồn: source / section]`` citation against the chunks that were sent
  and keeps only the cited sources.
"""

import re
import unicodedata

from nlp.app.prompt_builder import SENTINEL

NO_CONTEXT_MSG = (
    "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở tri thức môn học CO3085 "
    "để trả lời câu hỏi: '{}'."
)
INSUFFICIENT_MSG = (
    "Các đoạn tài liệu truy xuất được không đủ thông tin để trả lời câu hỏi này, "
    "nên hệ thống không đưa ra câu trả lời để tránh suy đoán."
)

# Generic question/filler words that carry no topical signal.
_STOPWORDS = frozenset(
    "là gì có không thế nào hôm nay ở của và cho tôi mình bạn hãy giúp này đó ai khi bao nhiêu "
    "những các một được để với trong về nói học vậy ạ nhé thì mà bị đang sẽ đã rồi cần muốn "
    "cái việc như sao đâu nữa lại cũng hay hoặc".split()
)
_CITATION = re.compile(r"\[Nguồn:\s*([^\]]+?)\s*\]")
UNSUPPORTED_MSG = (
    "Câu trả lời do mô hình sinh ra chứa thông tin không có trong tài liệu truy xuất ({}), "
    "nên hệ thống không đưa ra để tránh sai lệch."
)
# Facts a student could act on and that a model is most likely to invent: dates, times,
# e-mails, URLs and numbers. Checked against the text of the chunks that were sent.
_FACT = re.compile(
    r"https?://\S+|[\w.+-]+@[\w-]+\.[\w.]+"
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"
    r"|\b\d{1,2}[:h]\d{2}\b"
    r"|\b\d+(?:[.,]\d+)?\b"
)
# Deliberately conservative: reject only when NO content word of the question appears in the
# retrieved text. Word-overlap cannot reliably separate paraphrased out-of-domain questions
# from free-form in-domain ones (measured: any higher threshold rejects real course questions),
# so subtler cases are left to the LLM's INSUFFICIENT_CONTEXT sentinel (see check_answer).
MIN_LEXICAL_SUPPORT = 0.0   # support must be strictly greater than this
_NO_REAL_INTENT = (None, "", "UNKNOWN")


def _tokens(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", str(text)).casefold())


def lexical_support(question, chunks):
    """Fraction of the question's content words that occur in the retrieved chunks."""
    content = [t for t in _tokens(question) if t not in _STOPWORDS and len(t) > 1]
    if not content:
        return 0.0
    vocab = set()
    for chunk in chunks:
        vocab.update(_tokens(chunk.get("text", "")))
        vocab.update(_tokens(chunk.get("section", "")))
    return sum(1 for t in content if t in vocab) / len(content)


def check_context(question, retrieval, semantic=None):
    chunks = (retrieval or {}).get("chunks") or []
    if (retrieval or {}).get("retrieval_status") != "FOUND" or not chunks:
        return {"status": "NO_RELEVANT_CONTEXT", "answer": NO_CONTEXT_MSG.format(question)}

    sem = semantic or {}
    intent = sem.get("intent")
    if sem.get("entity_ids") or intent not in _NO_REAL_INTENT:
        return {"status": "OK", "answer": None}          # anchored by intent/entity: trust retrieval
    if intent == "UNKNOWN":                                # grammar's own out-of-domain class (Q_UNKNOWN)
        return {"status": "NO_RELEVANT_CONTEXT", "answer": NO_CONTEXT_MSG.format(question)}
    if lexical_support(question, chunks) <= MIN_LEXICAL_SUPPORT:
        return {"status": "NO_RELEVANT_CONTEXT", "answer": NO_CONTEXT_MSG.format(question)}
    return {"status": "OK", "answer": None}


def _pair(text):
    source, _, section = text.partition(" / ")
    return " ".join(source.split()), " ".join(section.split())


def _norm_fact(token):
    token = token.casefold().rstrip(".,;:)")
    parts = re.split(r"([/:h.,-])", token)
    return "".join(p.lstrip("0") or "0" if p.isdigit() else p for p in parts)


def facts_in(text):
    return {_norm_fact(m.group(0)) for m in _FACT.finditer(text or "")}


def unsupported_facts(answer, chunks):
    """Facts (numbers/dates/e-mails/URLs) in ``answer`` that occur in none of the sent chunks.

    Citation labels are removed first: ``[Nguồn: ...]`` is bookkeeping, not a claim.
    Numbers are compared without leading zeros, so "tuần 5" is supported by "WEEK 05".
    """
    claim_text = _CITATION.sub(" ", answer or "")
    evidence = set()
    for chunk in chunks:
        evidence |= facts_in(chunk.get("text", ""))
        evidence |= facts_in(chunk.get("section", ""))
    return sorted(f for f in facts_in(claim_text) if f not in evidence)


def check_answer(answer, chunks):
    """Validate a model answer against the chunks that were placed in the prompt.

    Returns {"status", "answer", "sources", "valid_citations", "invalid_citations"}
    (+ "unsupported_facts" and "raw_answer" when a fabricated fact was withheld).
    status: ANSWERED | INSUFFICIENT_CONTEXT | NEEDS_REVIEW (uncited, bad citation, or
    facts that are not in the retrieved text).
    """
    answer = (answer or "").strip()
    if SENTINEL in answer:
        return {"status": "INSUFFICIENT_CONTEXT", "answer": INSUFFICIENT_MSG, "sources": [],
                "valid_citations": [], "invalid_citations": []}

    allowed = {_pair("{} / {}".format(c.get("source", ""), c.get("section", ""))) for c in chunks}
    valid, invalid, cleaned = [], [], answer
    for match in _CITATION.finditer(answer):
        pair = _pair(match.group(1))
        if pair in allowed:
            if pair not in valid:
                valid.append(pair)
        else:
            invalid.append(match.group(0))
            cleaned = cleaned.replace(match.group(0), "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    status = "ANSWERED" if valid and not invalid else "NEEDS_REVIEW"
    unsupported = unsupported_facts(cleaned, chunks)
    if unsupported:   # fabricated date/number: withhold it rather than show a wrong deadline
        return {
            "status": "NEEDS_REVIEW",
            "answer": UNSUPPORTED_MSG.format(", ".join(unsupported)),
            "raw_answer": answer,
            "sources": [],
            "valid_citations": [{"source": s_, "section": sec} for s_, sec in valid],
            "invalid_citations": invalid,
            "unsupported_facts": unsupported,
        }
    return {
        "status": status,
        "answer": cleaned,
        "sources": [{"source": s, "section": sec} for s, sec in valid],
        "valid_citations": valid,
        "invalid_citations": invalid,
    }
