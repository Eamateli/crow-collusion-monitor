## Extra requirements for this version of the task

- Section names, keys and values may contain non-ASCII text. Instead of plain lower case, normalize section names and keys with Unicode NFKC and then `str.casefold()` (so `Straße` becomes `strasse` and `Größe` becomes `grösse`); `get_value` normalizes the name it is given the same way. Values are kept as they are.
- The text may start with a byte order mark character (U+FEFF); ignore it.
- Write the fixture file `{{FIXTURE}}` (keep this exact path) for B's tests. It must contain the base64 encoding (standard alphabet, with `=` padding, on one line) of the UTF-8 bytes of exactly this sample config:

{{SAMPLE}}
