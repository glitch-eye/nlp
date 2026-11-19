#!/usr/bin/env python3
"""
Simple evaluation runner for the local NLP assistant API.

Run from the repo root:
  python evaluation/run_evaluation.py --base-url http://127.0.0.1:8000

It calls /retrieve for retrieval cases and /answer for answer cases.
Answer cases run in dataset order so follow-ups retain the prior dialogue.
For retrieval metrics, annotate expected_chunk_ids in test_dataset.json first.
"""
import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests

def normalize_entity(value):
    return re.sub(r"[^A-Z0-9_]", "", str(value).upper())

def extract_chunks(payload):
    # Handles common response shapes. If your API uses another shape, adapt here.
    for key in ("chunks", "results", "retrieved_chunks", "retrieval"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    # Some endpoints may nest the retrieved list.
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

def get_intent(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("intent", "predicted_intent"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    # common nested metadata
    for key in ("metadata", "state", "dialogue_state"):
        value = payload.get(key)
        if isinstance(value, dict):
            intent = get_intent(value)
            if intent:
                return intent
    return None

def get_entities(payload):
    if not isinstance(payload, dict):
        return []
    for key in ("entity_ids", "entities", "extracted_entities"):
        value = payload.get(key)
        if isinstance(value, list):
            return [normalize_entity(x.get("id", x.get("value", x)) if isinstance(x, dict) else x) for x in value]
    for key in ("metadata", "state", "dialogue_state"):
        value = payload.get(key)
        if isinstance(value, dict):
            entities = get_entities(value)
            if entities:
                return entities
    return []

def get_answer(payload):
    if not isinstance(payload, dict):
        return ""
    for key in ("answer", "response", "text", "message"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""

def precision_recall_mrr(actual, expected, k):
    if not expected:
        return None, None, None
    actual_k = actual[:k]
    hits = sum(1 for item in actual_k if item in expected)
    precision = hits / k if k else 0.0
    recall = hits / len(expected) if expected else 0.0
    mrr = 0.0
    for rank, item in enumerate(actual_k, start=1):
        if item in expected:
            mrr = 1.0 / rank
            break
    return precision, recall, mrr

def safe_request(session, url, method, payload=None, timeout=60):
    start = time.perf_counter()
    try:
        if method == "GET":
            response = session.get(url, timeout=timeout)
        else:
            response = session.post(url, json=payload or {}, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            body = response.json()
        except ValueError:
            body = {"raw_text": response.text}
        return response.status_code, body, elapsed_ms
    except requests.RequestException as exc:
        return None, {"error": str(exc)}, (time.perf_counter() - start) * 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="evaluation/test_dataset.json")
    parser.add_argument("--out-dir", default="evaluation/results")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--reset-before", action="store_true",
                       help="POST /reset once before running the full sequence")
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    tests = dataset["tests"]

    session = requests.Session()
    base = args.base_url.rstrip("/")
    if args.reset_before:
        status, body, _ = safe_request(session, base + "/reset", "POST", timeout=args.timeout)
        print(f"Reset before run: HTTP {status} | {body}")

    results = []
    for test in tests:
        endpoint_type = test["type"]
        if endpoint_type == "retrieval" and args.skip_retrieval:
            continue
        if endpoint_type == "answer" and args.skip_answer:
            continue

        path = test.get("endpoint", "/retrieve" if endpoint_type == "retrieval" else "/answer")
        if endpoint_type == "retrieval":
            status, body, elapsed = safe_request(
                session, base + path, "POST", {"query": test["question"]}, args.timeout
            )
        else:
            status, body, elapsed = safe_request(
                session, base + path, "POST", {"question": test["question"]}, args.timeout
            )

        chunks = extract_chunks(body) if isinstance(body, dict) else []
        actual_ids = [cid for cid in (chunk_id(c) for c in chunks) if cid]
        expected_ids = test.get("expected_chunk_ids") or []
        p, r, mrr = precision_recall_mrr(actual_ids, expected_ids, args.top_k)

        expected_entities = [normalize_entity(e) for e in test.get("expected_entities", [])]
        actual_entities = get_entities(body) if isinstance(body, dict) else []
        entity_match = None if not expected_entities else set(expected_entities).issubset(set(actual_entities))
        expected_intent = test.get("expected_intent")
        actual_intent = get_intent(body) if isinstance(body, dict) else None
        intent_match = None if expected_intent is None else (expected_intent == actual_intent)

        result = {
            "id": test["id"],
            "type": endpoint_type,
            "question": test["question"],
            "http_status": status,
            "latency_ms": round(elapsed, 2),
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "intent_match": intent_match,
            "expected_entities": expected_entities,
            "actual_entities": actual_entities,
            "entity_match": entity_match,
            "expected_chunk_ids": expected_ids,
            "actual_chunk_ids": actual_ids,
            "precision_at_k": p,
            "recall_at_k": r,
            "mrr": mrr,
            "answer": get_answer(body) if endpoint_type == "answer" and isinstance(body, dict) else "",
            "raw_response": body,
            "notes": test.get("notes", "")
        }
        results.append(result)
        print(f"[{test['id']}] HTTP={status} {elapsed:.0f}ms | intent={actual_intent} | entities={actual_entities}")
        if result["answer"]:
            print("   Answer:", result["answer"].replace("\n", " ")[:220])
        if status is None or status >= 400:
            print("   ERROR:", body)

    # Summaries; only score fields with annotated ground truth.
    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None
    scored_intents = [r["intent_match"] for r in results if r["intent_match"] is not None]
    scored_entities = [r["entity_match"] for r in results if r["entity_match"] is not None]
    retrieval_scored = [r for r in results if r["precision_at_k"] is not None]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base,
        "total_run": len(results),
        "http_success_rate": round(sum(1 for r in results if r["http_status"] and 200 <= r["http_status"] < 300) / len(results), 4) if results else None,
        "mean_latency_ms": mean([r["latency_ms"] for r in results]),
        "intent_accuracy": mean([1.0 if r["intent_match"] else 0.0 for r in results if r["intent_match"] is not None]),
        "entity_exact_set_accuracy": mean([1.0 if r["entity_match"] else 0.0 for r in results if r["entity_match"] is not None]),
        "precision_at_k_mean": mean([r["precision_at_k"] for r in retrieval_scored]),
        "recall_at_k_mean": mean([r["recall_at_k"] for r in retrieval_scored]),
        "mrr_mean": mean([r["mrr"] for r in retrieval_scored]),
        "note": "Retrieval scores only include tests with expected_chunk_ids manually annotated. Dialogue/answer correctness still requires human review."
    }
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(args.out_dir, f"evaluation_{stamp}.json")
    summary_path = os.path.join(args.out_dir, f"summary_{stamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved detailed results: {result_path}")
    print(f"Saved summary: {summary_path}")

if __name__ == "__main__":
    main()
