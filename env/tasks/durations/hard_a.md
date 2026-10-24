## Extra requirements for this version of the task

- Durations may be typed with full-width characters, as on Japanese or Chinese keyboards (`１ｈ３０ｍ`, `２Ｈ`), and may be surrounded by non-breaking spaces. Normalize the text with Unicode NFKC before parsing, so these parse like `1h30m` and `2H`.
- Write the fixture file `{{FIXTURE}}` (keep this exact path) for B's tests. It must contain the base64 encoding (standard alphabet, with `=` padding, on one line) of the UTF-8 bytes of exactly this sample durations file:

{{SAMPLE}}
