"""Locate and spawn the compiled Swift helper binary."""

import logging
import os
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

HELPER_NAME = "BsafeCapture"


def find_helper() -> Path | None:
    """Look for the compiled Swift helper binary.

    Search order:
    1. BSAFE_HELPER_PATH env var (explicit override)
    2. Relative to the project source tree (development)
    3. System PATH
    """
    # 1. Explicit env var
    env_path = os.environ.get("BSAFE_HELPER_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        logger.warning("BSAFE_HELPER_PATH set but not found: %s", env_path)

    # 2. Check relative to the project source tree (development layout)
    project_root = Path(__file__).resolve().parent.parent
    candidate = project_root / "swift" / ".build" / "release" / HELPER_NAME
    if candidate.is_file():
        return candidate

    # 3. Fall back to PATH
    on_path = shutil.which(HELPER_NAME)
    if on_path:
        return Path(on_path)

    return None


def list_displays() -> list[dict]:
    """Query the Swift helper for connected displays."""
    import json

    helper = find_helper()
    if helper is None:
        raise FileNotFoundError(
            f"Swift helper '{HELPER_NAME}' not found. "
            "Build it with: cd swift && swift build -c release"
        )
    result = subprocess.run(
        [str(helper), "--list-displays"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list displays: {result.stderr.strip()}")
    # Output is JSONL: one JSON object per line.
    displays = []
    for line in result.stdout.strip().splitlines():
        if line:
            displays.append(json.loads(line))
    return displays


class StderrTail:
    """Drain a helper stderr pipe in the background, keeping the last lines."""

    def __init__(self, stream, max_lines: int = 50) -> None:
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, name="bsafe-helper-stderr", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        try:
            while True:
                try:
                    chunk = self._stream.readline(8192)
                except OSError, ValueError:
                    break
                if not chunk:
                    break
                line = chunk.decode("utf-8", errors="replace").rstrip("\r\n")
                with self._lock:
                    self._lines.append(line)
        finally:
            try:
                self._stream.close()
            except OSError, ValueError:
                pass

    def text(self, timeout: float = 1.0) -> str:
        self._thread.join(timeout)
        with self._lock:
            return "\n".join(self._lines)


def spawn_helper(socket_path: str, fps: int, display: str = "primary") -> subprocess.Popen:
    """Spawn the Swift helper, pointing it at our socket."""
    helper = find_helper()
    if helper is None:
        raise FileNotFoundError(
            f"Swift helper '{HELPER_NAME}' not found. "
            "Build it with: cd swift && swift build -c release"
        )
    logger.info("Spawning Swift helper: %s", helper)
    verbose = logger.isEnabledFor(logging.DEBUG)
    cmd = [str(helper), "--socket", socket_path, "--fps", str(fps)]
    # Always pass --display explicitly to avoid relying on Swift's implicit default.
    if display:
        cmd.extend(["--display", display])
    return subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.PIPE,
    )
