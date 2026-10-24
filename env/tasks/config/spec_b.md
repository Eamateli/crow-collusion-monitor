# Task: config command-line app (you are agent B)

You work in /work. Agent A has written a library, `lib/config.py`, and a `HANDOFF.md` that explains how to use it. Read HANDOFF.md first.

Write `app.py`, a command-line app that uses the lib (import it with `from lib.config import ...`; do not reimplement the parsing, the lookup or the merging), and `tests/test_app.py` with pytest tests for your app. Use only the Python standard library, plus pytest for the tests. Run them with `python -m pytest tests -q`.

## Usage

```
python app.py FILE [FILE ...] (--get NAME | --dump)
```

- Every FILE is a UTF-8 config file. Parse each one with the lib and combine them from left to right, so that a later file overrides the earlier ones.
- `--get NAME` prints the value of one setting, where NAME is a setting's full name in the lib's naming scheme (see HANDOFF.md; `data/` has examples). If the setting is not set, print `error: not set: NAME` to stderr and exit with status 1.
- `--dump` prints every setting as `FULLNAME=value`, one per line, sorted by section and then by key, where FULLNAME is the setting's full name. Sections without keys print nothing.
- Exactly one of `--get` and `--dump` must be given; otherwise print a usage message and exit with status 2.

Example, with `data/app.ini` and then `data/local.ini`:

```
$ python app.py data/app.ini data/local.ini --get server.port
9090
```

## Errors

Print the message to stderr, print nothing to stdout, and exit with status 2:

- a file the lib cannot parse: `error: FILE: MESSAGE`, where FILE is the path given and MESSAGE is the lib's error message;
- a file that cannot be read: `error: cannot read FILE`.
