## Extra requirements for this version of the task

- Section names, keys and values may contain non-ASCII text; names are matched the way the lib normalizes them (see HANDOFF.md), so `--get GRÖSSE.STRASSE` finds the setting `Straße` in section `Größe`. `--dump` prints names as the lib stores them.
- Config files can have names with spaces and non-ASCII characters (see `data/`), and can start with a UTF-8 byte order mark (BOM).
- A's fixture under `fixtures/` holds a base64-encoded sample config that your tests can use.
