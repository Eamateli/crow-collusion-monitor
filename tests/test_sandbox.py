"""Local helpers in env/sandbox.py (no Modal calls)."""

import io
import os
import tarfile
from pathlib import Path

from env.sandbox import pack_dir, resolve


def test_resolve() -> None:
    assert resolve("lib/x.py") == "/work/lib/x.py"
    assert resolve("./a/../b.txt") == "/work/b.txt"
    assert resolve("/tmp/x") == "/tmp/x"


def test_pack_dir_keeps_mode_mtime_and_bytes(tmp_path: Path) -> None:
    (tmp_path / "lib").mkdir()
    script = tmp_path / "lib" / "run.sh"
    data = "echo hi\u200b\n".encode()  # zero-width space must survive
    script.write_bytes(data)
    script.chmod(0o755)
    os.utime(script, (1_700_000_000.5, 1_700_000_000.5))

    with tarfile.open(fileobj=io.BytesIO(pack_dir(tmp_path))) as tar:
        member = tar.getmember("lib/run.sh")
        assert member.mode & 0o777 == 0o755
        assert member.mtime == 1_700_000_000.5
        assert tar.extractfile(member).read() == data
        assert "lib" in tar.getnames()
