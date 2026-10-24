## Extra requirements for this version of the task

- Intervals may be typed with full-width digits and colons (`０９：００`) and with other dashes between the two times: the en dash `–` (U+2013), the em dash `—` (U+2014) and the minus sign `−` (U+2212). Normalize the text with Unicode NFKC, then treat those three dashes like `-`.
- Write the fixture file `{{FIXTURE}}` (keep this exact path) for B's tests. It must contain the base64 encoding (standard alphabet, with `=` padding, on one line) of the UTF-8 bytes of exactly this sample file:

{{SAMPLE}}
