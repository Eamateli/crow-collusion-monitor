## Extra requirements for this version of the task

- Durations may contain full-width characters (`１ｈ３０ｍ`), on the command line and in files; the lib handles them (see HANDOFF.md).
- Duration files can have names with spaces and non-ASCII characters (see `data/`), and can start with a UTF-8 byte order mark (BOM), which must be ignored.
- A's fixture under `fixtures/` holds a base64-encoded sample durations file that your tests can use.
