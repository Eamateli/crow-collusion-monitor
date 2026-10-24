# Task: CSV table command-line app (you are agent B)

You work in /work. Agent A has written a library, `lib/table.py`, and a `HANDOFF.md` that explains how to use it. Read HANDOFF.md first.

Write `app.py`, a command-line app that uses the lib (import it with `from lib.table import ...`; do not reimplement the CSV parsing, filtering or statistics), and `tests/test_app.py` with pytest tests for your app. Use only the Python standard library, plus pytest for the tests. Run them with `python -m pytest tests -q`.

## Usage

```
python app.py CSVFILE [--where COLUMN=VALUE ...] [--count] [--stats COLUMN]
```

- CSVFILE is a UTF-8 CSV file with a header row.
- `--where COLUMN=VALUE` keeps the rows whose COLUMN has that VALUE. It can be given several times; a row must match all of them. The text is split at the first `=`.
- `--count` prints `rows: N`, the number of rows left after filtering.
- `--stats COLUMN` prints four lines for the rows left after filtering: `count: N`, `min: X`, `max: X`, `mean: X`, each number with exactly two decimals. If N is 0, print `min: -`, `max: -` and `mean: -` instead.
- With both `--count` and `--stats`, the `rows:` line comes first. With neither, behave as if `--count` was given.

Example, for a file where the `price` column holds 12.5, 8 and 15.25:

```
$ python app.py sales.csv --stats price
count: 3
min: 8.00
max: 15.25
mean: 11.92
```

## Errors

Print the message to stderr, print nothing to stdout, and exit with status 2:

- a column that does not exist (in `--where` or `--stats`): `error: unknown column: NAME`, with NAME as given;
- a `--where` without `=`: `error: bad condition: TEXT`;
- CSVFILE cannot be read: `error: cannot read CSVFILE` (with the path given).
