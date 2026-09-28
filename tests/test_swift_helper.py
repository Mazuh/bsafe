"""StderrTail bounded-drain tests (no network, no model, fast)."""

import logging
import os
import sys
import threading

import bsafe.swift_helper
from bsafe.swift_helper import StderrTail, spawn_helper


def _write_fake_helper(directory):
    """Create an executable fake helper that floods stderr, then exits 3."""
    script = directory / "fake-helper"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "for i in range(4000):\n"
        '    sys.stderr.write(f"line {i:05d} " + "x" * 56 + "\\n")\n'
        "    sys.stderr.flush()\n"
        'sys.stderr.write("FAKE-HELPER-DONE\\n")\n'
        "sys.stderr.flush()\n"
        "sys.exit(3)\n"
    )
    script.chmod(0o755)
    return script


def test_spawn_helper_drains_large_stderr(tmp_path, monkeypatch, caplog):
    helper_path = _write_fake_helper(tmp_path)
    monkeypatch.setattr(bsafe.swift_helper, "find_helper", lambda: helper_path)
    caplog.set_level(logging.INFO, logger="bsafe.swift_helper")
    proc = spawn_helper("/tmp/unused.sock", 30)
    assert proc.stderr is not None
    drain = StderrTail(proc.stderr)
    try:
        assert proc.wait(timeout=10) == 3
    except BaseException:
        proc.kill()
        proc.wait(timeout=10)
        raise
    tail = drain.text()
    assert isinstance(tail, str)
    assert "b'" not in tail
    lines = tail.splitlines()
    assert len(lines) <= 50
    assert lines[-1] == "FAKE-HELPER-DONE"


def test_stderr_tail_bounded_and_decodes():
    r, w = os.pipe()
    # Start draining before writing: the payload may exceed the initial pipe buffer.
    tail = StderrTail(os.fdopen(r, "rb"), max_lines=10)
    chunks = [f"line-{i:04d}\n".encode("utf-8") for i in range(200)]
    chunks.append(b"\xff\xfe invalid\n")
    chunks.append(b"y" * 20000)

    def _write():
        with os.fdopen(w, "wb") as writer:
            writer.write(b"".join(chunks))

    # Write from a daemon thread so a stalled drain fails the test instead of hanging it.
    writer_thread = threading.Thread(target=_write, daemon=True)
    writer_thread.start()
    writer_thread.join(timeout=5.0)
    assert not writer_thread.is_alive()
    text = tail.text(timeout=5.0)
    assert isinstance(text, str)
    lines = text.splitlines()
    assert len(lines) == 10
    assert all(len(line) <= 8192 for line in lines)
    assert lines[-4] == "\ufffd\ufffd invalid"
    assert lines[-3] == "y" * 8192
    assert lines[-2] == "y" * 8192
    assert lines[-1] == "y" * 3616


def test_stderr_tail_empty_stream():
    r, w = os.pipe()
    os.close(w)
    reader = os.fdopen(r, "rb")
    assert StderrTail(reader).text() == ""
