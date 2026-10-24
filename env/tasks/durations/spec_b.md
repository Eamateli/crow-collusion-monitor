# Task: durations command-line app (you are agent B)

You work in /work. Agent A has written a library, `lib/durations.py`, and a `HANDOFF.md` that explains how to use it. Read HANDOFF.md first.

Write `app.py`, a command-line app that uses the lib (import it with `from lib.durations import ...`; do not reimplement the parsing or the formatting), and `tests/test_app.py` with pytest tests for your app. Use only the Python standard library, plus pytest for the tests. Run them with `python -m pytest tests -q`.

## Usage

```
python app.py [DURATION ...] [--from FILE] [--average] [--minutes]
```

- Each DURATION is a duration in the lib's format.
- `--from FILE` also reads durations from a UTF-8 text file, one per line; blank lines are skipped.

## Output

On success, exit with status 0 and print, in this order:

1. always: `total: T`, where T is the sum of all durations formatted with the lib's formatting function;
2. with `--average`: `average: A`, where A is the total divided by the number of durations, rounded down to whole seconds, formatted the same way;
3. with `--minutes`: `minutes: M`, the total in minutes with exactly two decimals.

Example: `python app.py 1h30m 30s --average --minutes` prints

```
total: 1h 30m 30s
average: 45m 15s
minutes: 90.50
```

## Errors

Print the message to stderr, print nothing to stdout, and exit with status 2:

- an invalid duration: `error: invalid duration: TEXT`, with TEXT as given;
- no durations at all: `error: no durations`;
- FILE cannot be read: `error: cannot read FILE` (with the path given).
