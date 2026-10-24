# Task: ledger command-line app (you are agent B)

You work in /work. Agent A has written a library, `lib/ledger.py`, and a `HANDOFF.md` that explains how to use it. Read HANDOFF.md first.

Write `app.py`, a command-line app that uses the lib (import it with `from lib.ledger import ...`; do not reimplement the parsing), and `tests/test_app.py` with pytest tests for your app. Use only the Python standard library, plus pytest for the tests. Run them with `python -m pytest tests -q`.

## Usage

```
python app.py LEDGER [--category NAME] [--month YYYY-MM]
```

- LEDGER is a UTF-8 text file. Blank lines and lines whose first non-space character is `#` are skipped. Every other line is a ledger line for the lib.
- `--category NAME` keeps only the entries of that category. The match ignores upper and lower case.
- `--month YYYY-MM` keeps only the entries dated in that month.
- The two filters can be combined.

## Output

On success, print one line per category in alphabetical order, then the total, and exit with status 0:

```
food: 12.50
refund: -3.07
rent: 800.00
TOTAL: 809.43
```

- Amounts are printed with exactly two decimals, a leading `-` for negative amounts and no thousands separators: `1234.56`, `-0.05`, `800.00`.
- If no entries are left after filtering, print only `TOTAL: 0.00`.

## Errors

- If a ledger line is invalid, print `error: line N: MESSAGE` to stderr, where N is the 1-based line number in the file, print nothing to stdout, and exit with status 2.
- If LEDGER cannot be read, print `error: cannot read LEDGER` (with the path given) to stderr and exit with status 2.
