"""Hidden app tests: config."""

import pytest
from crowtest import out_lines, run_app, write

BASE = """\
# shared settings
name = demo

[server]
host = localhost
port = 8080

[db]
url = sqlite:///demo.db?mode=rw
"""
LOCAL = "; local overrides\n[Server]\nPort = 9090\n\n[log]\nlevel = debug\n"


@pytest.fixture
def files(tmp_path):
    return write(tmp_path / "app.ini", BASE), write(tmp_path / "local.ini", LOCAL)


def test_get_from_one_file(files):
    proc = run_app(files[0], "--get", "server.port")
    assert proc.returncode == 0
    assert out_lines(proc) == ["8080"]


def test_get_top_level_setting(files):
    assert out_lines(run_app(files[0], "--get", "name")) == ["demo"]


def test_later_file_overrides(files):
    assert out_lines(run_app(*files, "--get", "server.port")) == ["9090"]
    assert out_lines(run_app(*files, "--get", "SERVER.host")) == ["localhost"]
    assert out_lines(run_app(files[1], files[0], "--get", "server.port")) == ["8080"]


def test_get_missing_setting(files):
    proc = run_app(*files, "--get", "server.nope")
    assert proc.returncode == 1
    assert proc.stderr.strip() == "error: not set: server.nope"
    assert proc.stdout == ""


def test_dump_sorted(files):
    assert out_lines(run_app(*files, "--dump")) == [
        "db.url=sqlite:///demo.db?mode=rw",
        "default.name=demo",
        "log.level=debug",
        "server.host=localhost",
        "server.port=9090",
    ]


def test_dump_skips_empty_sections(tmp_path):
    path = write(tmp_path / "c.ini", "[empty]\n[a]\nk = v\n")
    assert out_lines(run_app(path, "--dump")) == ["a.k=v"]


def test_parse_error_message(tmp_path):
    path = write(tmp_path / "bad.ini", "[a]\nk = v\noops\n")
    proc = run_app(path, "--dump")
    assert proc.returncode == 2
    assert proc.stderr.startswith(f"error: {path}: line 3:")
    assert proc.stdout == ""


def test_missing_file(tmp_path):
    proc = run_app(tmp_path / "nope.ini", "--dump")
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: cannot read")


def test_requires_get_or_dump(files):
    for args in ([], ["--dump", "--get", "name"]):
        proc = run_app(files[0], *args)
        assert proc.returncode == 2
        assert "usage" in proc.stderr.lower()  # not just Python failing to start app.py
        assert proc.stdout == ""


@pytest.mark.hard
def test_hard_odd_file_name_bom_and_unicode_names(tmp_path):
    text = "[Gr\xf6\xdfe]\nStra\xdfe = Wei\xdf\nNiveau = \xe9lev\xe9\n"
    path = write(tmp_path / "Einstellungen (Entwurf) \N{EN DASH} v2.ini", text, bom=True)
    assert out_lines(run_app(path, "--get", "GR\xd6SSE.STRASSE")) == ["Wei\xdf"]


@pytest.mark.hard
def test_hard_dump_normalized_names(tmp_path):
    path = write(tmp_path / "c.ini", "[Gr\xf6\xdfe]\nStra\xdfe = Wei\xdf\n", bom=True)
    assert out_lines(run_app(path, "--dump")) == ["gr\xf6sse.strasse=Wei\xdf"]
