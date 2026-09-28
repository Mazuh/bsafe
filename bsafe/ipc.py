"""Unix domain socket server: receives frames from Swift helper into a mailbox."""

import logging
import os
import queue
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable

from bsafe.protocol import (
    CMD_SHUTDOWN,
    MSG_DISPLAY_INFO,
    MSG_FRAME_RAW,
    MSG_STATS,
    DisplayInfo,
    pack_cmd_censor,
    pack_cmd_censor_seq,
    pack_cmd_censor_style,
    pack_cmd_request_frame,
    pack_cmd_start,
    pack_message,
    parse_display_info_payload,
    parse_raw_frame_payload,
    read_message_into,
)

logger = logging.getLogger(__name__)


class FrameMailbox:
    """Bounded latest-frame per-display mailbox.

    Keeps at most one pending frame per display. A busy display replaces its
    own pending frame instead of queuing behind (or ahead of) quiet displays,
    so a busy monitor cannot starve a quiet one. Consumers take pending
    displays in first-pending order (fair round-robin across displays).

    Compatible subset of :class:`queue.Queue` used by live code:
    ``get(block=True, timeout=None)``, ``get_nowait()``, ``put_nowait()``,
    ``empty()``, ``qsize()``. ``get`` returns ``(meta, pixels, receipt_mono)``
    where ``meta`` is a RawFrameMeta (or any object with ``display_id``),
    ``pixels`` is a memoryview over raw BGRA bytes (or legacy JPEG bytes),
    and ``receipt_mono`` is a local ``time.monotonic()`` timestamp taken on
    receipt (never the cross-process Swift clock).
    """

    def __init__(self, maxsize: int = 10):
        self.maxsize = maxsize
        self._latest: dict[int, tuple] = {}
        self._order: deque[int] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        # Counters for --stats (no per-frame warnings).
        self.replaced = 0
        self.dropped_full = 0

    def _store(self, meta, jpeg_data: bytes, receipt_mono: float) -> None:
        display_id = meta.display_id
        with self._not_empty:
            if display_id in self._latest:
                self._latest[display_id] = (meta, jpeg_data, receipt_mono)
                self.replaced += 1
                logger.debug("Frame replaced: display=%d (latest-frame mailbox)", display_id)
                return
            if self.maxsize > 0 and len(self._latest) >= self.maxsize:
                self.dropped_full += 1
                logger.debug(
                    "Frame dropped: display=%d mailbox at capacity %d",
                    display_id,
                    self.maxsize,
                )
                return
            self._latest[display_id] = (meta, jpeg_data, receipt_mono)
            self._order.append(display_id)
            self._not_empty.notify()

    def put_nowait(self, item) -> None:
        """Store ``(meta, pixels)`` (or ``(meta, pixels, receipt)``), replacing."""
        meta, jpeg_data = item[0], item[1]
        if len(item) > 2:
            receipt_mono = float(item[2])
        else:
            receipt_mono = time.monotonic()
        self._store(meta, jpeg_data, receipt_mono)

    def put(self, item, block: bool = True, timeout=None) -> None:
        """Non-blocking store; never blocks (drops only on display cap)."""
        self.put_nowait(item)

    def get(self, block: bool = True, timeout=None):
        """Return next ``(meta, pixels, receipt_mono)`` in fair display order."""
        with self._not_empty:
            if not block:
                if not self._latest:
                    raise queue.Empty
            elif timeout is None:
                while not self._latest:
                    self._not_empty.wait()
            else:
                end = time.monotonic() + timeout
                while not self._latest:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._not_empty.wait(remaining)
            while self._order:
                display_id = self._order.popleft()
                item = self._latest.pop(display_id, None)
                if item is not None:
                    return item
            raise queue.Empty

    def get_nowait(self):
        return self.get(block=False)

    def peek(self, display_id: int):
        """Return latest pending ``(meta, pixels, receipt_mono)`` without consuming.

        Returns None when no frame is pending for the display. Never reorders,
        removes, or counts (no replaced/dropped accounting).
        """
        with self._lock:
            return self._latest.get(display_id)

    def empty(self) -> bool:
        with self._lock:
            return not self._latest

    def qsize(self) -> int:
        with self._lock:
            return len(self._latest)

    def __len__(self) -> int:
        return self.qsize()


class FrameServer:
    """Binds a Unix socket, accepts one client, sends CMD_START, reads frames into a queue."""

    def __init__(
        self,
        socket_path: str,
        fps: int = 60,
        maxsize: int = 10,
        blur: float = 0.0,
        pixels: float = 0.0,
        censor_text: str | None = None,
        scale_percent: int = 100,
        persist_passes: int = 8,
        smooth_percent: int = 50,
        stats: bool = False,
        on_stats: Callable[[str], None] | None = None,
    ):
        self.socket_path = socket_path
        self.fps = fps
        self.blur = blur
        self.pixels = pixels
        self.censor_text = censor_text
        self.scale_percent = scale_percent
        self.persist_passes = persist_passes
        self.smooth_percent = smooth_percent
        self.stats = stats
        self.on_stats = on_stats
        self.frame_queue: FrameMailbox = FrameMailbox(maxsize=maxsize)
        self.displays: dict[int, DisplayInfo] = {}
        self.display_events: queue.Queue[DisplayInfo] = queue.Queue()
        self._sock: socket.socket | None = None
        self._client: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()

    def start(self):
        """Bind socket and start the reader thread."""
        # Remove stale socket file
        if os.path.exists(self.socket_path):
            logger.warning("Removing stale socket: %s", self.socket_path)
            os.unlink(self.socket_path)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.socket_path)
        self._sock.listen(1)
        self._sock.settimeout(30.0)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Accept a client and read frames until stopped."""
        try:
            logger.info("Waiting for Swift helper to connect...")
            self._client, _ = self._sock.accept()
            logger.info("Swift helper connected")

            # Send CMD_START with capture config
            self._send_to_client(
                pack_cmd_start(
                    self.fps,
                    self.scale_percent,
                    self.persist_passes,
                    self.smooth_percent,
                    self.stats,
                )
            )

            # Send censor style config if blur, pixels, or text is set
            if self.blur > 0.0 or self.pixels > 0.0 or self.censor_text is not None:
                self._send_to_client(
                    pack_cmd_censor_style(self.blur, self.pixels, self.censor_text)
                )

            # Read frames (payload lands in one preallocated buffer per message)
            recv_into = self._client.recv_into
            malformed_last_log: dict[int, float] = {}
            while not self._stop_event.is_set():
                try:
                    msg_type, payload = read_message_into(recv_into)
                except ConnectionError, OSError:
                    logger.info("Swift helper disconnected")
                    break
                except ValueError as e:
                    # The stream is desynchronized after a bad length header; drop the client.
                    logger.error("Invalid message framing from Swift helper: %s", e)
                    break

                try:
                    if msg_type == MSG_FRAME_RAW:
                        meta, pixels_view = parse_raw_frame_payload(payload)
                        logger.debug(
                            "Frame received: display=%d %dx%d seq=%d",
                            meta.display_id,
                            meta.width,
                            meta.height,
                            meta.seq,
                        )
                        self.frame_queue.put_nowait((meta, pixels_view))
                    elif msg_type == MSG_DISPLAY_INFO:
                        info = parse_display_info_payload(payload)
                        self.displays[info.display_id] = info
                        self.display_events.put(info)
                        logger.debug(
                            "Display info: display=%d capture=%dx%d points=%dx%d",
                            info.display_id,
                            info.capture_width,
                            info.capture_height,
                            info.points_width,
                            info.points_height,
                        )
                    elif msg_type == MSG_STATS:
                        text = bytes(payload).decode("utf-8")
                        if self.on_stats is not None:
                            self.on_stats(text)
                        else:
                            logger.info("%s", text)
                    else:
                        # Legacy MSG_FRAME (JPEG) and unknown types are ignored.
                        logger.debug("Ignoring message type 0x%02x", msg_type)
                except ValueError, struct.error:
                    now = time.monotonic()
                    last = malformed_last_log.get(msg_type, float("-inf"))
                    if now - last >= 5.0:
                        logger.warning("Skipping malformed message type 0x%02x", msg_type)
                        malformed_last_log[msg_type] = now
                    if msg_type == MSG_FRAME_RAW and len(payload) >= 4:
                        try:
                            (recredit_id,) = struct.unpack_from("!I", payload, 0)
                        except struct.error:
                            continue
                        try:
                            self.request_frame(recredit_id)
                        except OSError:
                            pass
                    continue

        except TimeoutError:
            logger.error("Timed out waiting for Swift helper to connect")
        except Exception:
            logger.exception("Error in frame server")
        finally:
            self._cleanup_client()

    def _send_to_client(self, data: bytes):
        """Thread-safe send to the connected client."""
        with self._write_lock:
            if self._client:
                self._client.sendall(data)

    def request_frame(self, display_id: int):
        """Grant one frame credit for a display (at most one outstanding)."""
        self._send_to_client(pack_cmd_request_frame(display_id))

    def send_censor_seq(
        self,
        display_id: int,
        frame_width: int,
        frame_height: int,
        seq: int,
        boxes: list[tuple[int, int, int, int]],
    ):
        """Send CMD_CENSOR_SEQ boxes (in capture pixels of frame ``seq``) to Swift."""
        self._send_to_client(pack_cmd_censor_seq(display_id, frame_width, frame_height, seq, boxes))

    def send_censor(
        self,
        display_id: int,
        frame_width: int,
        frame_height: int,
        boxes: list[tuple[int, int, int, int]],
    ):
        """Send CMD_CENSOR with overlay box coordinates to Swift."""
        self._send_to_client(pack_cmd_censor(display_id, frame_width, frame_height, boxes))

    def shutdown(self):
        """Send CMD_SHUTDOWN and stop the server."""
        self._stop_event.set()
        try:
            self._send_to_client(pack_message(CMD_SHUTDOWN))
        except OSError:
            pass
        self._cleanup_client()
        self._cleanup_socket()
        if self._thread:
            self._thread.join(timeout=3)

    def _cleanup_client(self):
        if self._client:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None

    def _cleanup_socket(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
