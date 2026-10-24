"""Workspaces as tar bytes: read members without extracting them, and write tars.

Workspace tars come from agent sandboxes, so they are untrusted: they are only ever read with
`tarfile` in memory and never extracted to disk here.
"""

import io
import math
import posixpath
import tarfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Member:
    path: str  # relative POSIX path, without a leading "./"
    kind: str  # file | dir | symlink | other
    mode: int  # permission bits
    mtime: float
    data: bytes = b""  # file content; empty for anything but a regular file
    link: str | None = None  # symlink or hard link target


def normalize(name: str) -> str:
    path = posixpath.normpath(name.lstrip("/"))
    return "" if path == "." else path


def read_tar(data: bytes) -> list[Member]:
    """Members in archive order. A path that appears twice keeps its last entry, as tar would."""
    members: dict[str, Member] = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            path = normalize(m.name)
            if not path:
                continue
            if m.isreg():
                kind, content = "file", tar.extractfile(m).read()
            else:
                kind = "dir" if m.isdir() else "symlink" if m.issym() else "other"
                content = b""
            link = m.linkname if (m.issym() or m.islnk()) else None
            members.pop(path, None)
            members[path] = Member(path, kind, m.mode & 0o7777, m.mtime, content, link)
    return list(members.values())


def write_tar(members: Iterable[Member]) -> bytes:
    """POSIX tar of regular files and directories (other kinds are not written)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for m in sorted(members, key=lambda m: m.path):
            info = tarfile.TarInfo(m.path)
            info.mode, info.mtime = m.mode, m.mtime
            if m.kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif m.kind == "file":
                info.size = len(m.data)
                tar.addfile(info, io.BytesIO(m.data))
    return buf.getvalue()


def pack_tree(files: dict[str, bytes], mtime: float) -> bytes:
    """{relative path: bytes} -> tar bytes, every file mode 0644 with one mtime."""
    return write_tar(Member(path, "file", 0o644, mtime, data) for path, data in files.items())


def byte_entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte (CLAUDE.md section 7, equation 1)."""
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())
