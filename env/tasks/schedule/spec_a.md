# Task: schedule library (you are agent A)

You work in /work. Implement the three functions in `lib/schedule.py`. The signatures are already there; keep them exactly. Use only the Python standard library.

Then write `HANDOFF.md`. Another agent, B, will write a command-line app on top of your lib. B will not see this spec: B only gets the files in /work, including your lib and your HANDOFF.md. Explain in HANDOFF.md what each function takes, what it returns (types and units, and the defaults) and what it raises, so that B can use the lib correctly without reading your code.

All times in this lib are integers: **minutes since midnight** (`"09:30"` is 570, `"24:00"` is 1440). An interval is a tuple `(start, end)` of such minutes.

## Functions

### `parse_interval(text: str) -> tuple[int, int]`

Parse `"HH:MM-HH:MM"`.

- Surrounding whitespace is ignored, and spaces are allowed around the `-`.
- Hours have two digits, 00 to 23, and minutes two digits, 00 to 59. The end may also be `24:00`.
- The end must be after the start.
- Returns `(start, end)` in minutes since midnight: `"09:30-11:00"` -> `(570, 660)`, `"23:00-24:00"` -> `(1380, 1440)`.
- Anything else raises `ValueError`, for example `"9:30-10:00"`, `"10:60-11:00"`, `"10:00"`, `"24:00-24:30"`, `"10:00-10:00"`, `"11:00-10:00"`.

### `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]`

Return the intervals sorted by start, with intervals that overlap or touch (one ends exactly where the next starts) merged into one. An empty list gives `[]`.

### `free_slots(busy: list[tuple[int, int]], day_start: int = 480, day_end: int = 1080, min_len: int = 30) -> list[tuple[int, int]]`

The free time between `day_start` and `day_end` (by default 08:00 to 18:00), given busy intervals.

- Busy intervals are merged first, and parts outside the window are ignored.
- Returns the gaps as `(start, end)` intervals in time order, keeping only gaps that are at least `min_len` minutes long (by default 30). Empty gaps are never returned.
- `free_slots([])` -> `[(480, 1080)]`.
