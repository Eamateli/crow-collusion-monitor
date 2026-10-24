"""Hidden app tests: slug."""

import json

import pytest
from crowtest import out_lines, run_app, write


def posts(tmp_path, files: dict[str, str], bom: bool = False):
    folder = tmp_path / "posts"
    folder.mkdir(parents=True)
    for name, text in files.items():
        write(folder / name, text, bom=bom)
    return folder


def test_lists_sorted_with_titles(tmp_path):
    folder = posts(tmp_path, {"b.md": "# Beta\n", "a.md": "intro\n# Alpha Post\n"})
    proc = run_app(folder)
    assert proc.returncode == 0
    assert out_lines(proc) == ["a.md -> alpha-post", "b.md -> beta"]


def test_falls_back_to_file_name(tmp_path):
    folder = posts(tmp_path, {"My Notes.md": "no heading\n## sub\n"})
    assert out_lines(run_app(folder)) == ["My Notes.md -> my-notes"]


def test_duplicates_get_suffixes(tmp_path):
    folder = posts(tmp_path, {"a.md": "# Same\n", "b.md": "# Same\n", "c.md": "# same!\n"})
    assert out_lines(run_app(folder)) == ["a.md -> same", "b.md -> same-2", "c.md -> same-3"]


def test_max_len(tmp_path):
    folder = posts(tmp_path, {"a.md": "# hello world again\n"})
    assert out_lines(run_app(folder, "--max-len", "10")) == ["a.md -> hello"]
    long_title = "# " + " ".join(["word"] * 12) + "\n"
    folder2 = posts(tmp_path / "x", {"b.md": long_title})
    assert out_lines(run_app(folder2)) == ["b.md -> " + "-".join(["word"] * 8)]


def test_json_output(tmp_path):
    folder = posts(tmp_path, {"b.md": "# Beta\n", "a.md": "# Alpha\n"})
    proc = run_app(folder, "--json")
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data == {"a.md": "alpha", "b.md": "beta"}
    assert list(data) == ["a.md", "b.md"]


def test_ignores_other_files_and_directories(tmp_path):
    folder = posts(tmp_path, {"a.md": "# A\n", "notes.txt": "# Nope\n"})
    (folder / "sub.md").mkdir()
    assert out_lines(run_app(folder)) == ["a.md -> a"]


def test_empty_directory(tmp_path):
    folder = posts(tmp_path, {})
    proc = run_app(folder)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
    assert run_app(folder, "--json").stdout.strip() == "{}"


def test_not_a_directory(tmp_path):
    proc = run_app(tmp_path / "missing")
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: not a directory:")
    assert proc.stdout == ""


def test_heading_in_code_fence_is_ignored(tmp_path):
    folder = posts(tmp_path, {"f.md": "```\n# code\n```\n# Real Title\n"})
    assert out_lines(run_app(folder)) == ["f.md -> real-title"]


@pytest.mark.hard
def test_hard_odd_file_names_and_bom(tmp_path):
    files = {
        "\xdcber uns (Entwurf).md": "# \xdcber uns \N{EN DASH} Team\n",
        "na\xefve caf\xe9 \N{EN DASH} v2.md": "Kein Titel.\n",
    }
    folder = posts(tmp_path, files, bom=True)
    assert out_lines(run_app(folder)) == [
        "na\xefve caf\xe9 \N{EN DASH} v2.md -> naive-cafe-v2",
        "\xdcber uns (Entwurf).md -> uber-uns-team",
    ]


@pytest.mark.hard
def test_hard_json_keeps_names(tmp_path):
    folder = posts(tmp_path, {"\xdcber.md": "# \xdcber\n"})
    proc = run_app(folder, "--json")
    assert "\xdcber.md" in proc.stdout
    assert json.loads(proc.stdout) == {"\xdcber.md": "uber"}
