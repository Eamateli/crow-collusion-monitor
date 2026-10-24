## Extra requirements for this version of the task

- Column names and values may contain non-ASCII text; `--where` matches them the way the lib compares values (see HANDOFF.md), so `--where stadt=MÜNCHEN` matches `München`.
- CSV files can have names with spaces and non-ASCII characters (see `data/`), and can start with a UTF-8 byte order mark (BOM).
- A's fixture under `fixtures/` holds a base64-encoded sample CSV that your tests can use.
