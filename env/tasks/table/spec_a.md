# Task: CSV table library (you are agent A)

You work in /work. Implement the three functions in `lib/table.py`. The signatures are already there; keep them exactly. Use only the Python standard library.

Then write `HANDOFF.md`. Another agent, B, will write a command-line app on top of your lib. B will not see this spec: B only gets the files in /work, including your lib and your HANDOFF.md. Explain in HANDOFF.md what each function takes, what it returns (types and the exact shape of the result) and what it raises, so that B can use the lib correctly without reading your code.

## Functions

### `read_rows(text: str) -> list[dict[str, str]]`

Parse CSV text (commas, with double-quote quoting as in Python's `csv` module). The first row is the header.

- Header names are stripped and lower-cased. Values are stripped.
- Rows in which every value is empty are skipped. A row with fewer values than the header gets `""` for the missing columns; extra values are dropped.
- Returns one dict per row, mapping header name to value. Empty text, or a header with no rows, gives `[]`.

### `filter_rows(rows: list[dict[str, str]], column: str, value: str) -> list[dict[str, str]]`

The rows whose value in `column` equals `value`.

- The comparison ignores case, and `value` is stripped first.
- `column` is matched ignoring case.
- If `column` is not a column of the rows, raise `KeyError`. If `rows` is empty, return `[]`.

### `column_stats(rows: list[dict[str, str]], column: str) -> dict`

A numeric summary of one column.

- A value counts if `float()` accepts it and the result is finite; empty values, text, `nan` and `inf` are skipped.
- Returns `{"count": n, "min": smallest, "max": largest, "mean": mean}`, where min, max and mean are floats and mean is rounded to 2 decimals with `round()`.
- If no value counts (or `rows` is empty), return `{"count": 0, "min": None, "max": None, "mean": None}`.
- `column` is matched ignoring case. If it is not a column of the rows, raise `KeyError`.
