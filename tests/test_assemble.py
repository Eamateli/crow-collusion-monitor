"""The assembled workspace keeps each agent's raw output, whatever the defense did."""

from crow.assemble import assemble
from crow.workspace import Member, write_tar
from crow.wrap import Snapshot, snapshot

T0, T1, T2, T3, SEALED = 1000.0, 2000.0, 3000.0, 4000.0, 500.0


def snap(files: dict[str, tuple[str, float] | tuple[str, float, int]]) -> Snapshot:
    members = []
    for path, spec in files.items():
        text, mtime, mode = spec if len(spec) == 3 else (*spec, 0o644)
        members.append(Member(path, "file", mode, mtime, text.encode()))
    return snapshot(write_tar(members))


STARTER = {"lib/x.py": ("stub\n", T0), "data/d.txt": ("d\n", T0)}


def text(assembled, path):
    return assembled.snapshot.entries[path].text


def test_each_agents_changes_are_laid_over_the_starter_with_authors():
    starter = snap(STARTER)
    a = snap({**STARTER, "lib/x.py": ("by a\n", T1), "HANDOFF.md": ("note\n", T1)})
    b = snap(
        {
            "lib/x.py": ("by a\n", T1),
            "data/d.txt": ("d\n", T0),
            "HANDOFF.md": ("note\n", T1),
            "data/cases.json": ("{}\n", T2),
        }
    )
    c = snap(
        {
            "lib/x.py": ("by c\n", T3),
            "data/d.txt": ("d\n", T0),
            "HANDOFF.md": ("note\n", T1),
            "data/cases.json": ("{}\n", T2),
            "app.py": ("app\n", T3),
        }
    )
    result = assemble(starter, [("A", starter, a), ("B", a, b), ("C", b, c)])
    assert result.authors == {
        "lib/x.py": ["A", "C"],
        "HANDOFF.md": ["A"],
        "data/cases.json": ["B"],
        "app.py": ["C"],
    }
    assert text(result, "lib/x.py") == "by c\n"
    assert "data/d.txt" not in result.authors  # still the starter's


def test_sealing_does_not_change_what_an_earlier_agent_wrote():
    starter = snap(STARTER)
    raw_a = snap(
        {**STARTER, "lib/x.py": ("code  # comment\n", T1), "notes/extra.txt": ("hidden\n", T1)}
    )
    # The sealed view: the comment is stripped, the extra file dropped, everything re-stamped.
    view_b = snap({"lib/x.py": ("code\n", SEALED), "data/d.txt": ("d\n", SEALED)})
    raw_b = snap(
        {
            "lib/x.py": ("code\n", SEALED),
            "data/d.txt": ("d\n", SEALED),
            "data/cases.json": ("{}\n", T2),
        }
    )
    result = assemble(starter, [("A", starter, raw_a), ("B", view_b, raw_b)])
    assert text(result, "lib/x.py") == "code  # comment\n"
    assert text(result, "notes/extra.txt") == "hidden\n"
    assert result.authors["lib/x.py"] == ["A"]
    assert result.snapshot.entries["data/d.txt"].mtime == T0  # the seal's stamp is not the agent's
    assert text(result, "data/cases.json") == "{}\n"


def test_a_later_edit_on_a_sealed_view_replaces_the_file():
    starter = snap(STARTER)
    raw_a = snap({**STARTER, "lib/x.py": ("code  # comment\n", T1)})
    view_b = snap({"lib/x.py": ("code\n", SEALED), "data/d.txt": ("d\n", SEALED)})
    raw_b = snap({"lib/x.py": ("code\nmore\n", T2), "data/d.txt": ("d\n", SEALED)})
    result = assemble(starter, [("A", starter, raw_a), ("B", view_b, raw_b)])
    assert text(result, "lib/x.py") == "code\nmore\n"
    assert result.authors["lib/x.py"] == ["A", "B"]


def test_a_metadata_only_change_keeps_the_raw_content():
    starter = snap(STARTER)
    raw_a = snap({**STARTER, "lib/x.py": ("code  # comment\n", T1)})
    view_b = snap({"lib/x.py": ("code\n", SEALED), "data/d.txt": ("d\n", SEALED)})
    raw_b = snap({"lib/x.py": ("code\n", SEALED, 0o755), "data/d.txt": ("d\n", SEALED)})
    result = assemble(starter, [("A", starter, raw_a), ("B", view_b, raw_b)])
    entry = result.snapshot.entries["lib/x.py"]
    assert entry.text == "code  # comment\n" and entry.mode == 0o755


def test_deletes_and_renames_are_tracked():
    starter = snap(STARTER)
    a = snap({"lib/x.py": ("stub\n", T0), "data/moved.txt": ("d\n", T0)})
    b = snap({"data/moved.txt": ("d\n", T0)})
    result = assemble(starter, [("A", starter, a), ("B", a, b)])
    assert "lib/x.py" not in result.snapshot.entries and result.deleted["lib/x.py"] == "B"
    assert "data/d.txt" not in result.snapshot.entries and result.deleted["data/d.txt"] == "A"
    assert result.authors["data/moved.txt"] == ["A"]
