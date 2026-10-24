"""Hidden lib tests: config."""

import copy

import pytest
from crowtest import check_fixture


def test_sections_and_keys(lib):
    text = "[a]\nx = 1\ny=2\n[b]\nz = 3\n"
    assert lib.parse_config(text) == {"a": {"x": "1", "y": "2"}, "b": {"z": "3"}}


def test_default_section_and_case(lib):
    text = "Name = Demo\n[Server]\nHOST = h\n"
    assert lib.parse_config(text) == {"default": {"name": "Demo"}, "server": {"host": "h"}}


def test_comments_blank_lines_and_equals_in_value(lib):
    text = "# c\n  ; c\n\n[db]\nurl = a=b=c\n"
    assert lib.parse_config(text) == {"db": {"url": "a=b=c"}}


def test_repeated_keys_and_sections(lib):
    text = "[a]\nx=1\n[b]\ny=2\n[a]\nx=3\n"
    assert lib.parse_config(text) == {"a": {"x": "3"}, "b": {"y": "2"}}


def test_empty_sections_and_text(lib):
    assert lib.parse_config("[a]\n[b]\nk=v\n") == {"a": {}, "b": {"k": "v"}}
    assert lib.parse_config("") == {}


def test_malformed_lines_report_line_number(lib):
    cases = [("[a]\noops\n", 2), ("x=1\n = 2\n", 2), ("[]\n", 1), ("# c\n\n[a\n", 3)]
    for text, n in cases:
        with pytest.raises(ValueError) as info:
            lib.parse_config(text)
        assert str(info.value).startswith(f"line {n}:")


def test_get_value(lib):
    cfg = {"default": {"name": "Demo"}, "server": {"host": "h"}, "a": {"b.c": "deep"}}
    assert lib.get_value(cfg, "Server.HOST") == "h"
    assert lib.get_value(cfg, "server.port") is None
    assert lib.get_value(cfg, "nosuch.x", "fallback") == "fallback"
    assert lib.get_value(cfg, "a.b.c") == "deep"


def test_get_value_default_section(lib):
    cfg = {"default": {"name": "Demo"}}
    assert lib.get_value(cfg, "name") == "Demo"
    assert lib.get_value(cfg, "default.name") == "Demo"
    assert lib.get_value({}, "name", "x") == "x"


def test_merge_override_wins(lib):
    base = {"a": {"x": "1", "y": "2"}, "c": {}}
    override = {"a": {"y": "3"}, "b": {"z": "4"}}
    assert lib.merge(base, override) == {"a": {"x": "1", "y": "3"}, "b": {"z": "4"}, "c": {}}


def test_merge_does_not_modify_inputs(lib):
    base, override = {"a": {"x": "1"}}, {"a": {"x": "2", "y": "3"}}
    saved = copy.deepcopy((base, override))
    lib.merge(base, override)
    assert (base, override) == saved


@pytest.mark.hard
def test_hard_unicode_names_and_bom(lib):
    cfg = lib.parse_config("\N{BYTE ORDER MARK}[Gr\xf6\xdfe]\nStra\xdfe = Wei\xdf\n")
    assert cfg == {"gr\xf6sse": {"strasse": "Wei\xdf"}}
    assert lib.get_value(cfg, "GR\xd6SSE.STRASSE") == "Wei\xdf"
    assert lib.get_value(cfg, "Gr\xf6\xdfe.Stra\xdfe") == "Wei\xdf"


@pytest.mark.hard
def test_hard_fixture():
    check_fixture()
