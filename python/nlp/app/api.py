"""Small HTTP API exposing only the Course Assistant retrieval stage."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nlp.app.assistant import CourseAssistant
from nlp.app.dialogue import DialogueState
from nlp.app.knowledge_base import KnowledgeBase, default_data_dir
from nlp.app.retriever import Retriever


class RetrievalService:
    """Load the course knowledge base, retrieval index, and CourseAssistant orchestrator once."""

    def __init__(self, data_dir=None, top_k=3, min_score=0.08, lexical_weight=0.75, min_lexical_score=0.0):
        self.data_dir = Path(data_dir or default_data_dir())
        self.knowledge_base = KnowledgeBase(self.data_dir)
        self.retriever = Retriever(
            self.knowledge_base.documents,
            aliases=self.knowledge_base.aliases,
            top_k=top_k,
            min_score=min_score,
            lexical_weight=lexical_weight,
            min_lexical_score=min_lexical_score,
        )
        self.assistant = CourseAssistant(
            knowledge_base=self.knowledge_base,
            parser=self.retriever.parser,
            retriever=self.retriever,
            top_k=top_k,
        )
        self.dialogue_state = DialogueState()

    def retrieve(self, query, use_dialogue=False, dialogue_state=None):
        if not use_dialogue and dialogue_state is None:
            return self.retriever.retrieve(query)
        state = DialogueState.from_dict(dialogue_state) if dialogue_state is not None else self.dialogue_state
        resolution = self.assistant.dialogue_manager.resolve(query, state)
        resolved_query = resolution["resolved_query"]
        parse_tree = self.assistant.parser.parse(resolved_query) if self.assistant.parser else None
        semantic = self.assistant._build_semantic_frame(resolved_query, parse_tree, resolution)
        res = self.retriever.retrieve(resolved_query, semantic=semantic)
        res["original_query"] = query
        res["resolved_query"] = resolved_query
        res["resolution"] = resolution
        return res

    def answer(self, question, dialogue_state=None, reset=False):
        if reset:
            self.dialogue_state.reset()
        state = DialogueState.from_dict(dialogue_state) if dialogue_state is not None else self.dialogue_state
        return self.assistant.answer(question, dialogue_state=state)

    def reset_dialogue(self):
        self.dialogue_state.reset()
        return self.dialogue_state.to_dict()


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CourseAssistantRetrieval/1.0"

        def _send_json(self, status_code, payload):
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path != "/health":
                self._send_json(404, {"error": "not_found"})
                return
            self._send_json(200, {
                "status": "ok",
                "service": "retrieval",
                "document_count": len(service.knowledge_base.documents),
                "top_k": service.retriever.top_k,
                "min_score": service.retriever.min_score,
                "lexical_weight": service.retriever.lexical_weight,
                "semantic_weight": service.retriever.semantic_weight,
                "min_lexical_score": service.retriever.min_lexical_score,
            })

        def do_POST(self):
            if self.path not in ("/retrieve", "/answer", "/reset"):
                self._send_json(404, {"error": "not_found"})
                return
            if self.path == "/reset":
                self._send_json(200, {"status": "reset", "dialogue_state": service.reset_dialogue()})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid_content_length"})
                return
            if content_length <= 0 or content_length > 1024 * 1024:
                self._send_json(400, {"error": "request_body_size_invalid"})
                return
            try:
                request = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "invalid_json"})
                return
            if not isinstance(request, dict):
                self._send_json(400, {"error": "json_object_required"})
                return
            query = request.get("query") or request.get("question")
            if not isinstance(query, str) or not query.strip():
                self._send_json(400, {"error": "non_empty_query_required"})
                return
            if self.path == "/answer":
                self._send_json(
                    200,
                    service.answer(
                        query.strip(),
                        dialogue_state=request.get("dialogue_state"),
                        reset=bool(request.get("reset", False)),
                    ),
                )
                return
            self._send_json(
                200,
                service.retrieve(
                    query.strip(),
                    use_dialogue=bool(request.get("use_dialogue", False)),
                    dialogue_state=request.get("dialogue_state"),
                ),
            )

        def log_message(self, format_string, *args):
            # Keep the standard request log concise and avoid logging request bodies.
            super().log_message(format_string, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run the retrieval-only Course Assistant API")
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--lexical-weight", type=float, default=0.75)
    parser.add_argument("--min-lexical-score", type=float, default=0.0)
    args = parser.parse_args()

    service = RetrievalService(
        data_dir=args.data_dir,
        top_k=args.top_k,
        min_score=args.min_score,
        lexical_weight=args.lexical_weight,
        min_lexical_score=args.min_lexical_score,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(
        "Retrieval API listening at http://{}:{} ({} chunks; top_k={}; min_score={})".format(
            args.host,
            args.port,
            len(service.knowledge_base.documents),
            args.top_k,
            args.min_score,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping retrieval API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
