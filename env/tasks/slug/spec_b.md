# Task: slug command-line app (you are agent B)

You work in /work. Agent A has written a library, `lib/slug.py`, and a `HANDOFF.md` that explains how to use it. Read HANDOFF.md first.

Write `app.py`, a command-line app that uses the lib (import it with `from lib.slug import ...`; do not reimplement the slug rules or the title search), and `tests/test_app.py` with pytest tests for your app. Use only the Python standard library, plus pytest for the tests. Run them with `python -m pytest tests -q`.

## Usage

```
python app.py DIR [--max-len N] [--json]
```

For every file directly in DIR whose name ends with `.md` (no subdirectories, and nothing that is not a regular file), in sorted order of the file name:

- the title is the file's first level-1 heading, found with the lib; if it has none, the title is the file name without `.md`;
- the slug is the lib's slug of the title, made unique among the slugs already given in this run (the lib has a function for that).

Files are UTF-8. `--max-len N` is passed to the lib's slug function; without it, the lib's default is used.

## Output

On success, exit with status 0 and print one line per file, `FILENAME -> SLUG`, where FILENAME is the file name only:

```
hello-world.md -> hello-world
notes.md -> notes
second-post.md -> hello-world-2
```

- With `--json`, print instead one JSON object that maps each file name to its slug, in the same order, made with `json.dumps(..., ensure_ascii=False)`.
- If there are no `.md` files, print nothing (with `--json`, print `{}`).

## Errors

If DIR is not a directory, print `error: not a directory: DIR` (with the path given) to stderr, print nothing to stdout, and exit with status 2.
