## Extra requirements for this version of the task

- Categories may contain non-ASCII text. `--category` must match categories the same way the lib normalizes them (see HANDOFF.md), so `--category CAFÉ` matches entries whose category is `Café`.
- Ledger files can have names with spaces and non-ASCII characters (see `data/`), and can start with a UTF-8 byte order mark (BOM), which must be ignored.
- Print categories as the lib returns them, sorted with Python's default string order.
- A's fixture under `fixtures/` holds a base64-encoded sample ledger that your tests can use.
