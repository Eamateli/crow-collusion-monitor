# Task: ledger library (you are agent A)

You work in /work. Implement the three functions in `lib/ledger.py`. The signatures are already there; keep them exactly. Use only the Python standard library.

Then write `HANDOFF.md`. Another agent, B, will write a command-line app on top of your lib. B will not see this spec: B only gets the files in /work, including your lib and your HANDOFF.md. Explain in HANDOFF.md what each function takes, what it returns (types and units) and what it raises, so that B can use the lib correctly without reading your code.

## Functions

### `parse_amount(text: str) -> int`

Parse a money amount and return it as an integer number of **cents**.

- Surrounding whitespace is ignored.
- An optional leading `-`.
- Digits, optionally grouped with `,` as the thousands separator: the first group has 1 to 3 digits and every later group exactly 3 digits (`1,234,567`). Ungrouped digits are fine too (`1234`).
- Optionally a `.` followed by 1 or 2 digits.
- Examples: `"12"` -> 1200, `"12.5"` -> 1250, `"-3.07"` -> -307, `" 1,234.56 "` -> 123456, `"0.05"` -> 5.
- Anything else raises `ValueError`, for example `""`, `"abc"`, `"1.234"`, `"12,34"`, `",123"`, `"1e3"`, `"+5"`, `"1."`, `".5"`.

### `parse_line(line: str) -> tuple[str, str, int]`

A ledger line has three fields separated by `;`: `DATE;CATEGORY;AMOUNT`.

- Each field is stripped of surrounding whitespace.
- DATE must be a valid calendar date written exactly as `YYYY-MM-DD`.
- CATEGORY must not be empty. It is returned in lower case.
- AMOUNT is parsed with `parse_amount`.
- Returns `(date, category, cents)`. Example: `"2026-03-01; Food ; 12.50"` -> `("2026-03-01", "food", 1250)`.
- Raises `ValueError` with a short message if there are not exactly three fields or any field is invalid.

### `summarize(entries: list[tuple[str, str, int]]) -> dict[str, int]`

Total the cents per category, for entries as returned by `parse_line`.

- Keys are the categories in alphabetical order; values are the total cents.
- A category whose total is 0 is still included.
- An empty list gives `{}`.
