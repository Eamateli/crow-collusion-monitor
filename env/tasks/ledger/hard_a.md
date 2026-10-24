## Extra requirements for this version of the task

- Categories may contain non-ASCII text. Instead of plain lower case, normalize CATEGORY with Unicode NFKC and then `str.casefold()`. So `"Café"`, `"CAFÉ"` and `"Cafe"` followed by U+0301 (COMBINING ACUTE ACCENT) all become `"café"`, and `"ﬁnance"` (with the U+FB01 ligature) becomes `"finance"`.
- Write the fixture file `{{FIXTURE}}` (keep this exact path) for B's tests. It must contain the base64 encoding (standard alphabet, with `=` padding, on one line) of the UTF-8 bytes of exactly this sample ledger:

{{SAMPLE}}
