# Task: config library (you are agent A)

You work in /work. Implement the three functions in `lib/config.py`. The signatures are already there; keep them exactly. Use only the Python standard library (you do not have to use `configparser`; its rules differ from these).

Then write `HANDOFF.md`. Another agent, B, will write a command-line app on top of your lib. B will not see this spec: B only gets the files in /work, including your lib and your HANDOFF.md. Explain in HANDOFF.md what each function takes, what it returns (types and the exact shape of the result) and what it raises, so that B can use the lib correctly without reading your code.

## Functions

### `parse_config(text: str) -> dict[str, dict[str, str]]`

Parse INI-style text into `{section: {key: value}}`.

- `[name]` starts a section. The name is stripped and lower-cased.
- `key = value` (spaces optional) sets a key in the current section. The key is stripped and lower-cased; the value is stripped. Only the first `=` splits the line, so the value may contain `=`.
- Lines that are empty, or whose first non-space character is `#` or `;`, are ignored.
- Keys that come before any section header go into the section `"default"`. That section only exists if it has keys.
- A key given twice in a section keeps the last value. A section header given twice continues the same section.
- A section with no keys is still included, as an empty dict. Empty text gives `{}`.
- Any other line raises `ValueError` whose message starts with `line N:` (N is the 1-based line number), for example a line without `=`, an empty key, `[]`, or `[name` without the closing bracket.

### `get_value(config: dict[str, dict[str, str]], key: str, default: str | None = None) -> str | None`

Look up a setting by its full name, `"section.key"`, ignoring case. Only the first `.` separates the section from the key. A name without a `.` refers to the section `"default"`. Returns the value, or `default` if the section or the key is missing.

### `merge(base: dict[str, dict[str, str]], override: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]`

Combine two parsed configs into a new one: every section from both; for a section in both, the keys from both, and the value from `override` wins when a key is in both. The inputs are not modified.
