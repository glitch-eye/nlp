#!/usr/bin/env python3
"""Evaluation runner for the Course Assistant HTTP API (retrieval + parse + answer + dialogue).

Start the API first (in the terminal that has the LLM environment variables set):
    python -m nlp.app.api --host 127.0.0.1 --port 8000

Then, from the repo root (nlp/):
    python python/run_evaluation.py --reset-before
    python python/run_evaluation.py --require-real-model --timeout 180     # real-model run

Per-test fields in input/test_dataset.json (all optional except id/type/question):
    expected_intent / expected_intents   single intent, or the set of intents of a multi-intent question
    expected_entities                    entity IDs that must all be extracted
    expected_chunk_ids                   gold chunks; at least one must be in the top-k (hit@k), P/R/MRR also reported
    expected_status                      list of acceptable statuses, e.g. ["INSUFFICIENT_CONTEXT", "NO_RELEVANT_CONTEXT"]
    must_contain                         facts the answer must include; "a|b" = any alternative, every item required
    must_not_contain                     strings that must NOT appear in the answer
    resolved_query_contains              the dialogue-resolved query must contain this text
    expected_parse_status                PARSED | UNPARSED
    reset_before                         POST /reset before this test (isolates it from earlier dialogue)
    category                             grouping label for the summary

A test with none of the expectation fields is reported as UNLABELLED, never silently counted as a pass.
Retrieval cases call /retrieve, answer cases call /answer, both in dataset order so follow-ups keep dialogue state.
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
ROOT = Path(__file__).resolve().parent.parent  # repo root (nlp/)
DEFAULT_DATASET = ROOT / "input" / "test_dataset.json"
DEFAULT_OUT_DIR = ROOT / "evaluation" / "results"


# ----------------------------------------------------------------------------- response readers
def fold(text):
    return unicodedata.normalize("NFC", str(text)).casefold()


def normalize_entity(value):
    return re.sub(r"[^A-Z0-9_]", "", str(value).upper())


def extract_chunks(payload):
    for key in ("chunks", "results", "retrieved_chunks", "retrieval"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    for key in ("data", "result"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            found = extract_chunks(nested)
            if found:
                return found
    return []


def chunk_id(chunk):
    if not isinstance(chunk, dict):
        return str(chunk)
    for key in ("chunk_id", "id", "doc_id", "source_id"):
        if chunk.get(key) is not None:
            return str(chunk[key])
    return None


def get_ids(body):
    """Retrieved chunk IDs in rank order (answer responses carry retrieved_chunk_ids)."""
    if not isinstance(body, dict):
        return []
    ids = body.get("retrieved_chunk_ids")
    if isinstance(ids, list):
        return [str(i) for i in ids]
    return [cid for cid in (chunk_id(c) for c in extract_chunks(body)) if cid]


def get_intent(body):
    if not isinstance(body, dict):
        return None
    for key in ("intent", "predicted_intent"):
        if isinstance(body.get(key), str):
            return body[key]
    return None


def get_intents(body):
    """All intents: the list of a multi-intent response, else the single intent."""
    if not isinstance(body, dict):
        return []
    if isinstance(body.get("intents"), list) and body["intents"]:
        return [str(i) for i in body["intents"]]
    intent = get_intent(body)
    return [intent] if intent else []


def get_entities(body):
    if not isinstance(body, dict):
        return []
    for key in ("entity_ids", "extracted_entities"):
        value = body.get(key)
        if isinstance(value, list):
            return [normalize_entity(x.get("id", x.get("value", x)) if isinstance(x, dict) else x) for x in value]
    return []


def get_answer(body):
    if not isinstance(body, dict):
        return ""
    for key in ("answer", "response", "text", "message"):
        if isinstance(body.get(key), str):
            return body[key]
    return ""


def get_status(body):
    return body.get("status") if isinstance(body, dict) and isinstance(body.get("status"), str) else None


# ----------------------------------------------------------------------------- scoring helpers
def fact_present(answer, fact):
    """A fact is 'a|b|c': present if ANY alternative occurs (case-insensitive, NFC)."""
    haystack = fold(answer)
    return any(fold(alt.strip()) in haystack for alt in fact.split("|") if alt.strip())


def missing_facts(answer, facts):
    return [f for f in facts if not fact_present(answer, f)]


def forbidden_found(answer, forbidden):
    haystack = fold(answer)
    return [f for f in forbidden if fold(f) in haystack]


def precision_recall_mrr(actual, expected, k):
    """Precision@k uses k as the denominator (fewer than k results is penalised)."""
    if not expected:
        return None, None, None, None
    top = actual[:k]
    hits = sum(1 for item in top if item in expected)
    precision = hits / k if k else 0.0
    recall = len({i for i in top if i in expected}) / len(set(expected))
    mrr = next((1.0 / rank for rank, item in enumerate(top, 1) if item in expected), 0.0)
    return precision, recall, mrr, hits > 0


def is_model_real(model):
    text = fold(model or "")
    return bool(text) and "mock" not in text and "fallback" not in text


# ----------------------------------------------------------------------------- one test
def evaluate_test(test, http_status, body, elapsed_ms, top_k):
    body = body if isinstance(body, dict) else {}
    is_answer = test["type"] == "answer"
    ids = get_ids(body)
    answer = get_answer(body) if is_answer else ""
    status = get_status(body) if is_answer else None
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    http_ok = http_status is not None and 200 <= http_status < 300
    if not http_ok:
        check("http", False, "HTTP {} {}".format(http_status, str(body.get("error", ""))[:80]))

    expected_intent = test.get("expected_intent")
    expected_intents = test.get("expected_intents") or []
    actual_intent, actual_intents = get_intent(body), get_intents(body)
    intent_match = None
    if expected_intents:
        intent_match = set(actual_intents) == set(expected_intents)
        check("intents", intent_match, "expected {} got {}".format(sorted(expected_intents), sorted(actual_intents)))
    elif expected_intent is not None:
        intent_match = expected_intent == actual_intent
        check("intent", intent_match, "expected {} got {}".format(expected_intent, actual_intent))

    expected_entities = [normalize_entity(e) for e in test.get("expected_entities", [])]
    actual_entities = get_entities(body)
    entity_match = entity_exact = None
    if expected_entities:
        entity_match = set(expected_entities) <= set(actual_entities)
        entity_exact = set(expected_entities) == set(actual_entities)
        check("entities", entity_match, "expected {} got {}".format(expected_entities, actual_entities))

    if test.get("expected_parse_status"):
        got = body.get("parse_status")
        check("parse_status", got == test["expected_parse_status"], "expected {} got {}".format(test["expected_parse_status"], got))

    expected_ids = test.get("expected_chunk_ids") or []
    p, r, mrr, hit = precision_recall_mrr(ids, expected_ids, top_k)
    if expected_ids:
        check("retrieval", hit, "gold {} top-{} {}".format(expected_ids, top_k, ids[:top_k]))

    expected_status = test.get("expected_status") or []
    if expected_status:
        check("status", status in expected_status, "expected {} got {}".format(expected_status, status))

    facts_missing, forbidden = [], []
    must = test.get("must_contain") or []
    if must and status == "ANSWERED":
        facts_missing = missing_facts(answer, must)
        check("facts", not facts_missing, "missing {}".format(facts_missing) if facts_missing else "")
    if test.get("must_not_contain") and answer:
        forbidden = forbidden_found(answer, test["must_not_contain"])
        check("forbidden_text", not forbidden, "found {}".format(forbidden) if forbidden else "")

    # Language drift: the guard withholds CJK answers (language_drift) and the raw text may also show it.
    raw_text = answer + " " + str(body.get("raw_answer") or "")
    drift = is_answer and (bool(body.get("language_drift")) or (bool(_CJK.search(raw_text)) and not _CJK.search(test["question"])))
    if drift:
        check("language", False, "model answered in an unexpected language (CJK)")

    if test.get("resolved_query_contains"):
        resolved = body.get("resolved_query") or ""
        needle = test["resolved_query_contains"]
        check("resolved_query", fold(needle) in fold(resolved), "needs {!r}, resolved {!r}".format(needle, resolved))

    # passed: True/False if anything was checked, None if the test carries no expectations.
    expectation_checks = [c for c in checks if c["name"] != "http"]
    passed = None if (not expectation_checks and http_ok) else all(c["ok"] for c in checks)

    llm = body.get("llm") if isinstance(body.get("llm"), dict) else None
    return {
        "id": test["id"], "type": test["type"], "category": test.get("category", "uncategorised"),
        "question": test["question"], "http_status": http_status, "latency_ms": round(elapsed_ms, 2),
        "status": status, "parse_status": body.get("parse_status"),
        "expected_intent": expected_intent, "expected_intents": expected_intents,
        "actual_intent": actual_intent, "actual_intents": actual_intents, "intent_match": intent_match,
        "expected_entities": expected_entities, "actual_entities": actual_entities,
        "entity_match": entity_match, "entity_exact": entity_exact,
        "expected_chunk_ids": expected_ids, "actual_chunk_ids": ids,
        "precision_at_k": p, "recall_at_k": r, "mrr": mrr, "hit_at_k": hit,
        "expected_status": expected_status, "facts_missing": facts_missing, "forbidden_found": forbidden,
        "resolved_query": body.get("resolved_query"), "llm": llm, "language_drift": drift,
        "answer": answer, "checks": checks, "passed": passed,
        "raw_response": body, "notes": test.get("notes", ""),
    }


# ----------------------------------------------------------------------------- summary
def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _rate(num, den):
    return None if not den else round(num / den, 4)


def summarize(results, base_url, health, top_k):
    scored = [r for r in results if r["passed"] is not None]
    passed = [r for r in scored if r["passed"]]
    unlabelled = [r for r in results if r["passed"] is None]
    answers = [r for r in results if r["type"] == "answer"]
    latencies = sorted(r["latency_ms"] for r in results)

    def tests_with(name):
        return [r for r in results if any(c["name"] == name for c in r["checks"])]

    gold = [r for r in results if r["expected_chunk_ids"]]
    must_refuse = [r for r in answers if r["expected_status"] and "ANSWERED" not in r["expected_status"]]
    must_answer = [r for r in answers if r["expected_status"] == ["ANSWERED"]]
    with_facts = [r for r in results if any(c["name"] == "facts" for c in r["checks"])]
    wrong = [r for r in answers if r["status"] == "ANSWERED" and (r["facts_missing"] or r["forbidden_found"])]
    intent_tests = [r for r in results if r["intent_match"] is not None]
    entity_tests = [r for r in results if r["entity_match"] is not None]
    parse_tests = tests_with("parse_status")

    called = [r for r in answers if r["llm"] and r["llm"].get("called")]
    llm_known = any(r["llm"] for r in answers)
    models = sorted({str(r["llm"].get("model")) for r in called})
    fallback = [r for r in called if "fallback" in fold(r["llm"].get("model"))]
    mocked = [r for r in called if "mock" in fold(r["llm"].get("model"))]
    if not llm_known:
        verified = None
    else:
        verified = bool(called) and all(is_model_real(r["llm"].get("model")) for r in called)

    by_category = {}
    for r in scored:
        bucket = by_category.setdefault(r["category"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += 1 if r["passed"] else 0

    status_tests = tests_with("status")
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url, "top_k": top_k, "server": health,
        "total_run": len(results), "scored": len(scored), "passed": len(passed),
        "failed": len(scored) - len(passed), "unlabelled": len(unlabelled),
        "pass_rate": _rate(len(passed), len(scored)),
        "http_success_rate": _rate(sum(1 for r in results if r["http_status"] and 200 <= r["http_status"] < 300), len(results)),
        "mean_latency_ms": _mean(latencies),
        "p95_latency_ms": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
        "parser": {
            "intent_accuracy": _rate(sum(1 for r in intent_tests if r["intent_match"]), len(intent_tests)), "intent_n": len(intent_tests),
            "entity_recall": _rate(sum(1 for r in entity_tests if r["entity_match"]), len(entity_tests)),
            "entity_exact_set_accuracy": _rate(sum(1 for r in entity_tests if r["entity_exact"]), len(entity_tests)), "entity_n": len(entity_tests),
            "parse_status_accuracy": _rate(sum(1 for r in parse_tests if all(c["ok"] for c in r["checks"] if c["name"] == "parse_status")), len(parse_tests)),
            "unparsed_answer_rate": _rate(sum(1 for r in answers if r["parse_status"] == "UNPARSED"), len([r for r in answers if r["parse_status"]])),
        },
        "retrieval": {
            "labelled_n": len(gold),
            "hit_at_k": _mean([1.0 if r["hit_at_k"] else 0.0 for r in gold]),
            "precision_at_k_mean": _mean([r["precision_at_k"] for r in gold]),
            "recall_at_k_mean": _mean([r["recall_at_k"] for r in gold]),
            "mrr_mean": _mean([r["mrr"] for r in gold]),
        },
        "answers": {
            "status_accuracy": _rate(sum(1 for r in status_tests if all(c["ok"] for c in r["checks"] if c["name"] == "status")), len(status_tests)),
            "false_answer_rate": _rate(sum(1 for r in must_refuse if r["status"] == "ANSWERED"), len(must_refuse)), "must_refuse_n": len(must_refuse),
            "false_refusal_rate": _rate(sum(1 for r in must_answer if r["status"] != "ANSWERED"), len(must_answer)), "must_answer_n": len(must_answer),
            "fact_pass_rate": _rate(sum(1 for r in with_facts if not r["facts_missing"]), len(with_facts)), "fact_n": len(with_facts),
            "answered_but_wrong": len(wrong),
            "needs_review": sum(1 for r in answers if r["status"] == "NEEDS_REVIEW"),
            "language_drift": sum(1 for r in answers if r["language_drift"]),
        },
        "llm": {
            "server_reports": (health or {}).get("llm"), "answers_with_model_call": len(called),
            "models_seen": models, "fallback_answers": len(fallback), "mock_answers": len(mocked),
            "real_model_verified": verified,
        },
        "by_category": by_category,
        "note": ("Scores cover only tests that carry expectations; UNLABELLED tests are listed, not counted. "
                 "Gold chunk labels and facts were written from the KB text. Citation validity (answer_guard) only means the cited "
                 "source was sent in the prompt, not that it supports the claim."),
    }


def _fmt(v):
    return "n/a" if v is None else ("{:.0%}".format(v) if isinstance(v, float) and v <= 1 else str(v))


def format_report(summary, results):
    s, L = summary, []
    L.append("Course Assistant evaluation | {} | {} | top_k={}".format(s["created_at"], s["base_url"], s["top_k"]))
    llm = s["llm"]
    if llm["real_model_verified"] is True:
        L.append("LLM: REAL model verified: {}".format(", ".join(llm["models_seen"])))
    elif llm["real_model_verified"] is False:
        L.append("LLM: NOT a verified real-model run (mock answers: {}, fallback answers: {}, model calls: {}). "
                 "Results measure parser/retriever/guard only.".format(llm["mock_answers"], llm["fallback_answers"], llm["answers_with_model_call"]))
    else:
        L.append("LLM: UNKNOWN - the server did not report a model (restart it from the updated api.py).")
    L += ["", "tests: {} run | {} scored | {} passed | {} FAILED | {} unlabelled | pass rate {}".format(
        s["total_run"], s["scored"], s["passed"], s["failed"], s["unlabelled"], _fmt(s["pass_rate"])),
        "latency: mean {} ms, p95 {} ms | HTTP success {}".format(s["mean_latency_ms"], s["p95_latency_ms"], _fmt(s["http_success_rate"])), ""]
    p, rt, a = s["parser"], s["retrieval"], s["answers"]
    L += ["== Parser ==",
          "  intent accuracy        {} (n={})".format(_fmt(p["intent_accuracy"]), p["intent_n"]),
          "  entity recall          {} | exact-set {} (n={})".format(_fmt(p["entity_recall"]), _fmt(p["entity_exact_set_accuracy"]), p["entity_n"]),
          "  unparsed answer rate   {}".format(_fmt(p["unparsed_answer_rate"])),
          "== Retrieval (gold-labelled n={}) ==".format(rt["labelled_n"]),
          "  hit@{} {} | precision@{} {} | recall@{} {} | MRR {}".format(
              s["top_k"], _fmt(rt["hit_at_k"]), s["top_k"], _fmt(rt["precision_at_k_mean"]), s["top_k"], _fmt(rt["recall_at_k_mean"]), _fmt(rt["mrr_mean"])),
          "== Answers ==",
          "  status accuracy        {}".format(_fmt(a["status_accuracy"])),
          "  FALSE-ANSWER rate      {} (must refuse n={})".format(_fmt(a["false_answer_rate"]), a["must_refuse_n"]),
          "  false-refusal rate     {} (must answer n={})".format(_fmt(a["false_refusal_rate"]), a["must_answer_n"]),
          "  facts present          {} (n={}) | ANSWERED-but-wrong: {}".format(_fmt(a["fact_pass_rate"]), a["fact_n"], a["answered_but_wrong"]),
          "  NEEDS_REVIEW answers   {} | language drift: {}".format(a["needs_review"], a["language_drift"]),
          "== By category =="]
    L += ["  {:<20} {}/{}".format(c, b["passed"], b["total"]) for c, b in sorted(s["by_category"].items())]
    L += ["", "== Failures =="]
    failures = [r for r in results if r["passed"] is False]
    for r in failures:
        L.append("  [FAIL] {} ({}) {}".format(r["id"], r["category"], r["question"]))
        for c in r["checks"]:
            if not c["ok"]:
                L.append("         - {}: {}".format(c["name"], c["detail"]))
        if r["answer"]:
            L.append("         answer: {}".format(" ".join(r["answer"].split())[:140]))
    if not failures:
        L.append("  none")
    if s["unlabelled"]:
        L += ["", "== Unlabelled (no expectations, not scored) =="] + ["  {} {}".format(r["id"], r["question"]) for r in results if r["passed"] is None]
    L.append("")
    L.append(s["note"])
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- http + main
def safe_request(session, url, method, payload=None, timeout=60):
    start = time.perf_counter()
    try:
        response = session.get(url, timeout=timeout) if method == "GET" else session.post(url, json=payload or {}, timeout=timeout)
        elapsed = (time.perf_counter() - start) * 1000
        try:
            body = response.json()
        except ValueError:
            body = {"raw_text": response.text}
        return response.status_code, body, elapsed
    except requests.RequestException as exc:
        return None, {"error": str(exc)}, (time.perf_counter() - start) * 1000


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the Course Assistant API")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120, help="seconds per request; raise it (e.g. 180) for a local LLM")
    ap.add_argument("--reset-before", action="store_true", help="POST /reset once before the run")
    ap.add_argument("--skip-answer", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated test IDs (follow-up chains may then lose context)")
    ap.add_argument("--require-real-model", action="store_true", help="abort if the server reports a mock model; exit 3 if the run was not a verified real-model run")
    ap.add_argument("--fail-on-failure", action="store_true", help="exit 1 if any scored test failed")
    args = ap.parse_args(argv)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
    only = {i.strip() for i in args.only.split(",") if i.strip()}
    session, base = requests.Session(), args.base_url.rstrip("/")

    status, health, _ = safe_request(session, base + "/health", "GET", timeout=args.timeout)
    if status != 200:
        print("Cannot reach {}/health ({}).\nStart the API first:  python -m nlp.app.api --port 8000".format(base, health))
        return 2
    llm = health.get("llm") or {}
    print("Server: {} chunks, LLM provider={} model={}".format(health.get("document_count"), llm.get("provider"), llm.get("model")))
    if args.require_real_model and not is_model_real(llm.get("model")):
        print("--require-real-model: the server is configured with {!r}. Start the API with LLM_PROVIDER/LLM_MODEL/OPENAI_* set.".format(llm.get("model")))
        return 2
    if args.reset_before:
        status, body, _ = safe_request(session, base + "/reset", "POST", timeout=args.timeout)
        print("Reset before run: HTTP {}".format(status))

    results = []
    for test in dataset["tests"]:
        if (only and test["id"] not in only) or (test["type"] == "retrieval" and args.skip_retrieval) or (test["type"] == "answer" and args.skip_answer):
            continue
        if test.get("reset_before"):
            safe_request(session, base + "/reset", "POST", timeout=args.timeout)
        path = test.get("endpoint", "/retrieve" if test["type"] == "retrieval" else "/answer")
        key = "query" if test["type"] == "retrieval" else "question"
        http_status, body, elapsed = safe_request(session, base + path, "POST", {key: test["question"]}, args.timeout)
        result = evaluate_test(test, http_status, body, elapsed, args.top_k)
        results.append(result)
        mark = {True: "PASS", False: "FAIL", None: "----"}[result["passed"]]
        print("[{}] {:<4} {:<20} {:>6.0f}ms {}".format(mark, result["id"], result["status"] or result["type"], elapsed, test["question"]))
        for c in result["checks"]:
            if not c["ok"]:
                print("        - {}: {}".format(c["name"], c["detail"]))

    if not results:
        print("No tests ran (check --only / --skip-* flags).")
        return 2
    _, health_after, _ = safe_request(session, base + "/health", "GET", timeout=args.timeout)
    summary = summarize(results, base, health_after if isinstance(health_after, dict) else health, args.top_k)
    report = format_report(summary, results)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "evaluation_{}.json".format(stamp)).write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary_{}.json".format(stamp)).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report_{}.txt".format(stamp)).write_text(report, encoding="utf-8")
    print("\n" + report)
    print("Saved: {}".format(out / "report_{}.txt".format(stamp)))

    if args.require_real_model and summary["llm"]["real_model_verified"] is not True:
        print("--require-real-model: this run is NOT a verified real-model run.")
        return 3
    if args.fail_on_failure and summary["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
