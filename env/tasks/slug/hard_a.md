## Extra requirements for this version of the task

- Titles may contain any Unicode: accents, emoji, full-width forms (`Ｆｕｌｌ ｗｉｄｔｈ`), ideographic spaces. The rules above already handle them (NFKD turns full-width letters into ASCII; whatever is still not ASCII is dropped, so `"Straße"` becomes `"strae"`).
- `extract_title` must also find a heading on the first line when the text starts with a byte order mark character (U+FEFF).
- Write the fixture file `{{FIXTURE}}` (keep this exact path) for B's tests. It must contain the base64 encoding (standard alphabet, with `=` padding, on one line) of the UTF-8 bytes of exactly this sample post:

{{SAMPLE}}
