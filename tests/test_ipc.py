"""Integration test: mock client → server → queue (live v2 raw frames)."""

import io
import os
import socket
import struct
import tempfile
import time

import pytest

from bsafe.ipc import FrameServer
from bsafe.protocol import (
    CMD_CENSOR,
    CMD_CENSOR_SEQ,
    CMD_REQUEST_FRAME,
    CMD_START,
    MSG_FRAME,
    MSG_STATS,
    DisplayInfo,
    RawFrameMeta,
    pack_cmd_censor_seq,
    pack_cmd_request_frame,
    pack_display_info,
    pack_message,
    pack_raw_frame,
    parse_censor_payload,
    parse_censor_seq_payload,
    parse_cmd_start_payload,
    read_message,
)


def _raw_pixels(width=4, height=3):
    return bytes((i * 7) & 0xFF for i in range(width * height * 4))


def _split(raw: bytes):
    """Split a framed message into (msg_type, payload) for exact-bytes comparison."""
    return read_message(io.BytesIO(raw).read)


@pytest.fixture
def sock_path():
    """Provide a short socket path that fits within AF_UNIX limits."""
    fd, path = tempfile.mkstemp(prefix="bsafe-", suffix=".sock", dir="/tmp")
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _connect(sock_path):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    return client


def _read_start(client):
    msg_type, payload = read_message(client.recv)
    assert msg_type == CMD_START
    return parse_cmd_start_payload(payload)


def test_frame_server_receives_raw_frames(sock_path):
    server = FrameServer(sock_path, fps=3, maxsize=5)
    server.start()

    # Connect as mock Swift client
    client = _connect(sock_path)

    # Read CMD_START from server (new 6-byte payload)
    assert _read_start(client) == (3, 100, 8, 50, False)

    # Send display info + a raw frame
    info = DisplayInfo(1, 4, 3, 4, 3)
    t_sent = time.monotonic()
    client.sendall(pack_display_info(info))
    pixels = _raw_pixels(4, 3)
    meta = RawFrameMeta(1, 4, 3, 777, 42)
    client.sendall(pack_raw_frame(meta, pixels))

    # Display registry populated
    deadline = time.monotonic() + 2
    while 1 not in server.displays and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.displays[1] == info
    event = server.display_events.get(timeout=2)
    assert event == info

    # Wait for queue (mailbox returns meta, memoryview, local receipt monotonic)
    got_meta, view, receipt = server.frame_queue.get(timeout=2)
    t1 = time.monotonic()
    assert got_meta == meta
    assert isinstance(view, memoryview)
    assert bytes(view) == pixels
    assert t_sent <= receipt <= t1

    client.close()
    server.shutdown()


def test_frame_server_start_config(sock_path):
    server = FrameServer(
        sock_path, fps=10, scale_percent=150, persist_passes=4, smooth_percent=100, stats=True
    )
    server.start()
    client = _connect(sock_path)
    assert _read_start(client) == (10, 150, 4, 100, True)
    client.close()
    server.shutdown()


def test_frame_server_drops_when_full(sock_path):
    server = FrameServer(sock_path, fps=1, maxsize=2)
    server.start()

    client = _connect(sock_path)

    # Read CMD_START
    read_message(client.recv)

    # Send more frames than the queue can hold (distinct displays pin slots)
    sent = 8
    for display_id in range(100, 100 + sent):
        pixels = _raw_pixels(2, 2)
        client.sendall(pack_raw_frame(RawFrameMeta(display_id, 2, 2, 0, 1), pixels))

    # Wait for server thread to process all sent frames
    time.sleep(0.5)

    # Drain the queue and count — should be at most maxsize (2)
    received = 0
    while not server.frame_queue.empty():
        server.frame_queue.get_nowait()
        received += 1

    assert received <= 2
    assert received < sent  # some were dropped

    client.close()
    server.shutdown()


def test_frame_server_send_censor(sock_path):
    server = FrameServer(sock_path, fps=3)
    server.start()

    client = _connect(sock_path)

    # Read CMD_START
    read_message(client.recv)

    # Server sends legacy censor command (still supported)
    boxes = [(10, 20, 100, 200), (300, 400, 50, 60)]
    server.send_censor(1, 1920, 1080, boxes)

    # Client reads the censor command
    msg_type, payload = read_message(client.recv)
    assert msg_type == CMD_CENSOR
    display_id, fw, fh, parsed_boxes = parse_censor_payload(payload)
    assert display_id == 1
    assert fw == 1920
    assert fh == 1080
    assert parsed_boxes == boxes

    client.close()
    server.shutdown()


def test_frame_server_request_frame(sock_path):
    server = FrameServer(sock_path, fps=3)
    server.start()
    client = _connect(sock_path)
    read_message(client.recv)

    server.request_frame(7)
    msg_type, payload = read_message(client.recv)
    expected_type, expected_payload = _split(pack_cmd_request_frame(7))
    assert msg_type == expected_type == CMD_REQUEST_FRAME
    assert payload == expected_payload == struct.pack("!I", 7)

    client.close()
    server.shutdown()


def test_frame_server_send_censor_seq(sock_path):
    server = FrameServer(sock_path, fps=3)
    server.start()
    client = _connect(sock_path)
    read_message(client.recv)

    boxes = [(1, 2, 30, 40)]
    server.send_censor_seq(2, 800, 600, 55, boxes)
    msg_type, payload = read_message(client.recv)
    expected_type, expected_payload = _split(pack_cmd_censor_seq(2, 800, 600, 55, boxes))
    assert msg_type == expected_type == CMD_CENSOR_SEQ
    assert payload == expected_payload
    assert parse_censor_seq_payload(payload) == (2, 800, 600, 55, boxes)

    # A miss pass (zero boxes) round-trips too.
    server.send_censor_seq(2, 800, 600, 56, [])
    msg_type, payload = read_message(client.recv)
    assert msg_type == CMD_CENSOR_SEQ
    assert parse_censor_seq_payload(payload) == (2, 800, 600, 56, [])

    client.close()
    server.shutdown()


def test_frame_server_stats_callback(sock_path):
    received = []
    server = FrameServer(sock_path, fps=3, stats=True, on_stats=received.append)
    server.start()
    client = _connect(sock_path)
    fps, _, _, _, stats_flag = _read_start(client)
    assert fps == 3
    assert stats_flag is True

    client.sendall(pack_message(MSG_STATS, b"swift fps=60"))
    deadline = time.monotonic() + 2
    while not received and time.monotonic() < deadline:
        time.sleep(0.01)
    assert received == ["swift fps=60"]

    client.close()
    server.shutdown()


def test_frame_server_ignores_legacy_frame(sock_path):
    server = FrameServer(sock_path, fps=3)
    server.start()
    client = _connect(sock_path)
    read_message(client.recv)

    legacy = struct.pack("!IIIq", 1, 640, 480, 100) + b"\xff\xd8"
    client.sendall(pack_message(MSG_FRAME, legacy))
    time.sleep(0.3)
    assert server.frame_queue.empty()

    client.close()
    server.shutdown()


def test_frame_server_recredits_malformed_raw_frame(sock_path):
    from bsafe.protocol import (
        CMD_REQUEST_FRAME,
        MSG_FRAME_RAW,
        RAW_FRAME_META_FMT,
    )

    server = FrameServer(sock_path, fps=3)
    server.start()
    client = _connect(sock_path)
    read_message(client.recv)

    display_id = 9
    header = struct.pack(RAW_FRAME_META_FMT, display_id, 4, 3, 123, 7)
    short_pixels = b"\x00" * 10  # expected 4*3*4 = 48 bytes
    client.sendall(pack_message(MSG_FRAME_RAW, header + short_pixels))

    client.settimeout(2.0)
    msg_type, payload = read_message(client.recv)
    assert msg_type == CMD_REQUEST_FRAME
    assert payload == struct.pack("!I", display_id)

    assert server.frame_queue.empty()

    client.settimeout(0.3)
    try:
        extra_type, _extra_payload = read_message(client.recv)
    except OSError:
        extra_type = None
    assert extra_type != CMD_CENSOR_SEQ

    client.close()
    server.shutdown()


def test_frame_server_cleans_stale_socket(sock_path):
    # Create a stale socket file
    with open(sock_path, "w") as f:
        f.write("")

    server = FrameServer(sock_path, fps=1)
    server.start()

    # Should be able to connect
    client = _connect(sock_path)
    read_message(client.recv)

    client.close()
    server.shutdown()
