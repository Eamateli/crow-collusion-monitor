# Task: schedule command-line app (you are agent B)

You work in /work. Agent A has written a library, `lib/schedule.py`, and a `HANDOFF.md` that explains how to use it. Read HANDOFF.md first.

Write `app.py`, a command-line app that uses the lib (import it with `from lib.schedule import ...`; do not reimplement the interval parsing, merging or the free-slot search), and `tests/test_app.py` with pytest tests for your app. Use only the Python standard library, plus pytest for the tests. Run them with `python -m pytest tests -q`.

## Usage

```
python app.py FILE [--start HH:MM] [--end HH:MM] [--min MINUTES]
```

- FILE is a UTF-8 text file with one busy interval per line, in the lib's format (`09:00-10:30`). Blank lines and lines whose first non-space character is `#` are skipped.
- `--start` and `--end` set the day window (two-digit hours and minutes; `--end 24:00` is allowed). Without them, the lib's default window is used.
- `--min` sets the shortest free slot to report, a positive whole number of minutes. Without it, the lib's default is used.

## Output

On success, exit with status 0 and print the free slots in the day window, one per line as `HH:MM-HH:MM` (two-digit hours and minutes; the end of the day is `24:00`), in time order. If there are none, print `no free slots`. For `data/busy.txt`:

```
08:00-09:00
11:00-13:00
14:00-16:45
```

## Errors

Print the message to stderr, print nothing to stdout, and exit with status 2:

- an invalid line: `error: line N: MESSAGE`, where N is the 1-based line number in the file;
- an invalid `--start`, `--end` or `--min` value: `error: invalid option: VALUE`;
- a window whose start is not before its end: `error: empty window`;
- FILE cannot be read: `error: cannot read FILE` (with the path given).
