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
