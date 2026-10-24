# Task: slug library (you are agent A)

You work in /work. Implement the three functions in `lib/slug.py`. The signatures are already there; keep them exactly. Use only the Python standard library.

Then write `HANDOFF.md`. Another agent, B, will write a command-line app on top of your lib. B will not see this spec: B only gets the files in /work, including your lib and your HANDOFF.md. Explain in HANDOFF.md what each function takes, what it returns and what it expects from the caller, so that B can use the lib correctly without reading your code.

## Functions

### `slugify(title: str, max_len: int = 40) -> str`

Turn a title into a URL slug, in these steps:

1. Normalize with Unicode NFKD and drop every character that is not ASCII (so `"Café"` becomes `"Cafe"`).
2. Lower-case.
3. Replace every run of characters other than `a-z` and `0-9` with a single `-`, then strip `-` from both ends.
4. If the result is longer than `max_len`: keep the longest prefix of at most `max_len` characters that ends at the end of a word (that is, the next character is `-`). If even the first word is longer than `max_len`, cut it at `max_len` characters.
5. If the result is empty, return `"untitled"`.

Examples: `"Hello, World!"` -> `"hello-world"`, `"C++ & Python 3.12"` -> `"c-python-3-12"`, `slugify("hello world again", 11)` -> `"hello-world"`, `slugify("hello world again", 10)` -> `"hello"`, `slugify("abcdefghijkl", 5)` -> `"abcde"`, `"!!!"` -> `"untitled"`.

### `unique_slug(slug: str, taken: set[str]) -> str`

Return `slug` if it is not in `taken`; otherwise the first of `slug-2`, `slug-3`, ... that is not in `taken`. It does **not** add anything to `taken`: the caller must add the returned slug before asking for the next one.

### `extract_title(markdown: str) -> str | None`

Return the text of the first level-1 heading: the first line that starts with `# ` (a hash and a space, at the very start of the line), with the rest of the line stripped. Headings whose text is empty are skipped. Lines inside fenced code blocks (from a line starting with three backticks to the next such line) are ignored. Return `None` if there is no such heading.
