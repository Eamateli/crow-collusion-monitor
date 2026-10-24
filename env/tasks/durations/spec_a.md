# Task: durations library (you are agent A)

You work in /work. Implement the three functions in `lib/durations.py`. The signatures are already there; keep them exactly. Use only the Python standard library.

Then write `HANDOFF.md`. Another agent, B, will write a command-line app on top of your lib. B will not see this spec: B only gets the files in /work, including your lib and your HANDOFF.md. Explain in HANDOFF.md what each function takes, what it returns (types and units) and what it raises, so that B can use the lib correctly without reading your code.

## Functions

### `parse_duration(text: str) -> int`

Parse a duration and return it as a whole number of **seconds**.

- Surrounding whitespace is ignored and letters are case-insensitive.
- Either a plain non-negative integer, which means seconds (`"90"` -> 90), or one or more `<integer><unit>` parts with no spaces between them. Units: `d` (86400 s), `h` (3600 s), `m` (60 s), `s` (1 s). The parts must come in that order and each unit at most once.
- Examples: `"1h30m"` -> 5400, `"2d"` -> 172800, `"45s"` -> 45, `"1H5S"` -> 3605, `"90m"` -> 5400, `"1d2h3m4s"` -> 93784.
- Anything else raises `ValueError`, for example `""`, `"1x"`, `"30m1h"` (wrong order), `"1h1h"`, `"-5m"`, `"1.5h"`, `"1 h"`, `"h"`.

### `format_duration(seconds: int) -> str`

Format a number of seconds for people.

- Parts `<n>d`, `<n>h`, `<n>m`, `<n>s`, largest unit first, separated by single spaces, leaving out the parts that are 0: 5400 -> `"1h 30m"`, 3605 -> `"1h 5s"`, 59 -> `"59s"`, 90061 -> `"1d 1h 1m 1s"`.
- 0 -> `"0s"`.
- A negative number raises `ValueError`.

### `total(texts: list[str]) -> int`

The sum of `parse_duration` over all texts, in seconds. An empty list gives 0. If any text is invalid, the same `ValueError` is raised.
