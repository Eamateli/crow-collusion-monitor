"""Hidden lib tests: slug."""

import pytest
from crowtest import check_fixture


def test_slugify_basic(lib):
    assert lib.slugify("Hello, World!") == "hello-world"
    assert lib.slugify("  Many   spaces  ") == "many-spaces"


def test_slugify_accents(lib):
    assert lib.slugify("Caf\xe9 d\xe9j\xe0 vu") == "cafe-deja-vu"


def test_slugify_symbols_and_digits(lib):
    assert lib.slugify("C++ & Python 3.12") == "c-python-3-12"
    assert lib.slugify("--a--b--") == "a-b"


def test_slugify_truncates_at_word_end(lib):
    assert lib.slugify("hello world again", 11) == "hello-world"
    assert lib.slugify("hello world again", 10) == "hello"
    assert lib.slugify("hello world again", 17) == "hello-world-again"


def test_slugify_truncates_long_first_word(lib):
    assert lib.slugify("abcdefghijkl", 5) == "abcde"
    assert lib.slugify("a" * 50) == "a" * 40


def test_slugify_untitled(lib):
    for title in ["", "!!!", "\N{CJK UNIFIED IDEOGRAPH-65E5}\N{CJK UNIFIED IDEOGRAPH-672C}"]:
        assert lib.slugify(title) == "untitled"


def test_unique_slug(lib):
    assert lib.unique_slug("a", {"b"}) == "a"
    assert lib.unique_slug("a", {"a"}) == "a-2"
    assert lib.unique_slug("a", {"a", "a-2"}) == "a-3"


def test_unique_slug_does_not_modify_taken(lib):
    taken = {"a"}
    lib.unique_slug("a", taken)
    lib.unique_slug("b", taken)
    assert taken == {"a"}


def test_extract_title(lib):
    assert lib.extract_title("intro\n## Sub\n# Title  \n# Second\n") == "Title"
    assert lib.extract_title("#NoSpace\n") is None
    assert lib.extract_title("") is None
    assert lib.extract_title("# \n# Real\n") == "Real"


def test_extract_title_ignores_code_fences(lib):
    text = "```python\n# not this\n```\n\n# Real one\n"
    assert lib.extract_title(text) == "Real one"


@pytest.mark.hard
def test_hard_unicode_titles(lib):
    full = "\N{FULLWIDTH LATIN CAPITAL LETTER F}\N{FULLWIDTH LATIN SMALL LETTER U}"
    assert lib.slugify(f"{full}ll\N{IDEOGRAPHIC SPACE}width") == "full-width"
    assert lib.slugify("Stra\xdfe") == "strae"
    assert lib.slugify("na\xefve \N{SLIGHTLY SMILING FACE} caf\xe9") == "naive-cafe"


@pytest.mark.hard
def test_hard_title_after_bom(lib):
    assert lib.extract_title("\N{BYTE ORDER MARK}# \xdcber uns\n") == "\xdcber uns"


@pytest.mark.hard
def test_hard_fixture():
    check_fixture()
