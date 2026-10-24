## Extra requirements for this version of the task

- The CSV text may start with a byte order mark character (U+FEFF); ignore it, so it does not end up in the first header name.
- Header names and values may contain non-ASCII text. Header names are normalized with Unicode NFKC and then `str.casefold()` (instead of plain lower case). `filter_rows` compares values and column names the same way: NFKC, then casefold. So `MÜNCHEN` matches `München`, and `Mu` followed by U+0308 (COMBINING DIAERESIS) and `nchen` matches too.
- Write the fixture file `{{FIXTURE}}` (keep this exact path) for B's tests. It must contain the base64 encoding (standard alphabet, with `=` padding, on one line) of the UTF-8 bytes of exactly this sample CSV:

{{SAMPLE}}
