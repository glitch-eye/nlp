# Grammar v5: multi-intent query processing

## Design

The existing CFG remains the validator for one intent at a time. The system does
not blindly rewrite every occurrence of "và" as a conjunction. Instead,
`multi_intent.py` searches for a partition whose individual clauses are all
accepted by the existing CFG. A lexical phrase such as `CFG và Parsing` and
the comparison `Chương 3 và Chương 4 khác nhau như thế nào?` therefore remain
single clauses when the grammar recognizes them as one construction.

## Pipeline

```text
User query
   |
   v
Clause splitter / CFG validation
   |
   +---- one clause ------> legacy Parser -> Semantic Frame -> Retriever -> Answer
   |
   +---- many clauses ---> parse each clause independently
                              |
                              v
                      Semantic frame per clause
                              |
                              v
                      Retrieve / answer per clause
                              |
                              v
                    Merge evidence + answers + entities
                              |
                              v
                       Combined response
```

## Compatibility contract

Single-intent responses keep the legacy fields:
- `intent`
- `entities`
- `entity_ids`
- `parse_status`

Compound responses add:
- `intent: MULTI_INTENT`
- `intents: [...] `
- `clauses: [{query, intent, entity_ids, status, answer, sources}, ...]`

The existing single-intent tree shape is unchanged. The parser's explicit
`parse_multi()` / `parse_line()` representation uses an `S_MULTI` root with
one `CLAUSE` child per independently parsed clause.

## Example

Input:

`Chương 4 nói về gì và Tuần 5 học gì?`

Expected clause-level intent shape:

```json
{
  "intent": "MULTI_INTENT",
  "clauses": [
    {"intent": "GET_TOPIC", "entity_ids": ["CH04"]},
    {"intent": "GET_SCHEDULE", "entity_ids": ["WEEK_05"]}
  ]
}
```

## Validation

Run from the `python/` directory:

```bash
python -m unittest discover -s test -v
```

The new regression cases are in `test/test_multi_intent.py`.


## Generation and evaluation semantics

Multi-intent is intentionally implemented as **per-clause execution**. For a query
with multiple validated clauses, the assistant runs the existing single-intent
path for each clause and merges the resulting answers and evidence. This means
the system can issue one generation call per eligible clause; it does not
currently aggregate all clauses into one LLM prompt or guarantee one API
request per user question. A pre-generation Answer Guard may stop an individual
clause before `LLMClient.generate()`.

`LLMClient.call_history` records one privacy-safe metadata entry for each
actual `generate()` invocation. Evaluation slices the history around each
`assistant.answer()` call, so a multi-intent sample can report multiple model
calls and can be classified as `real_api`, `fallback`, or `mixed`. A row with
zero generation calls may have been stopped by the guard; this is different
from a missing telemetry record.

To run the relevant tests from `python/`:

~~~bash
python -m unittest discover -s test -p "test_multi_intent*.py" -v
python -m unittest discover -s test -p "test_answer_evaluation.py" -v
~~~

For a unified report:

~~~bash
python -m nlp.app.unified_evaluation --provider mock --max-samples 10
~~~

Use `--provider openai --require-real-model` only when the goal is to require
successful API generation for every evaluated row. Rows blocked before
generation intentionally fail that requirement.
