## Extra requirements for this version of the task

- Lines may use full-width digits and other dashes (`０９：００–１０：００`, `12:00 — 12:30`); the lib handles them (see HANDOFF.md).
- Files can have names with spaces and non-ASCII characters (see `data/`), and can start with a UTF-8 byte order mark (BOM), which must be ignored.
- A's fixture under `fixtures/` holds a base64-encoded sample file that your tests can use.
