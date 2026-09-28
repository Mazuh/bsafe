"""Unit tests for message framing round-trips."""

import io
import struct

from bsafe.protocol import (
    CENSOR_BOX_FMT,
    CENSOR_BOX_SIZE,
    CENSOR_HEADER_FMT,
    CENSOR_HEADER_SIZE,
    CENSOR_SEQ_HEADER_FMT,
    CENSOR_SEQ_HEADER_SIZE,
    CMD_CENSOR,
    CMD_CENSOR_SEQ,
    CMD_CENSOR_STYLE,
    CMD_REQUEST_FRAME,
    CMD_SHUTDOWN,
    CMD_START,
    CMD_STOP,
    DISPLAY_INFO_FMT,
    DISPLAY_INFO_SIZE,
    FRAME_META_FMT,
    FRAME_META_SIZE,
    HEADER_FMT,
    MAX_MESSAGE_SIZE,
    MSG_DISPLAY_INFO,
    MSG_FRAME,
    MSG_FRAME_RAW,
    MSG_STATS,
    RAW_FRAME_META_FMT,
    RAW_FRAME_META_SIZE,
    START_FMT,
    DisplayInfo,
    FrameMetadata,
    RawFrameMeta,
    pack_cmd_censor,
    pack_cmd_censor_seq,
    pack_cmd_censor_style,
    pack_cmd_request_frame,
    pack_cmd_start,
    pack_display_info,
    pack_message,
    pack_raw_frame,
    parse_censor_payload,
    parse_censor_seq_payload,
    parse_censor_style_payload,
    parse_cmd_start_payload,
    parse_display_info_payload,
    parse_frame_payload,
    parse_raw_frame_payload,
    read_message,
    read_message_into,
)


def _make_recv(data: bytes):
    """Create a recv_fn from bytes."""
    stream = io.BytesIO(data)
    return stream.read


def _make_recv_into(data: bytes, chunk: int = 1):
    """Create a recv_into(view, nbytes) fake returning at most chunk bytes per call."""

    stream = io.BytesIO(data)

    def recv_into(view, nbytes):
        want = min(chunk, nbytes)
        piece = stream.read(want)
        if not piece:
            return 0
        view[: len(piece)] = piece
        return len(piece)

    return recv_into


def test_pack_and_read_empty_payload():
    raw = pack_message(CMD_STOP)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_STOP
    assert payload == b""


def test_pack_and_read_with_payload():
    data = b"hello"
    raw = pack_message(CMD_START, data)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_START
    assert payload == data


def test_pack_cmd_start():
    raw = pack_cmd_start(fps=3)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_START
    assert payload == struct.pack(START_FMT, 3, 100, 8, 50, 0)
    assert parse_cmd_start_payload(payload) == (3, 100, 8, 50, False)


def test_frame_round_trip():
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg-data"
    meta = FrameMetadata(display_id=1, width=1920, height=1080, timestamp_ns=123456789)
    frame_payload = (
        struct.pack(FRAME_META_FMT, meta.display_id, meta.width, meta.height, meta.timestamp_ns)
        + jpeg
    )

    raw = pack_message(MSG_FRAME, frame_payload)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == MSG_FRAME

    parsed_meta, parsed_jpeg = parse_frame_payload(payload)
    assert parsed_meta == meta
    assert parsed_jpeg == jpeg


def test_parse_frame_payload_too_short():
    import pytest

    with pytest.raises(ValueError, match="too short"):
        parse_frame_payload(b"short")


def test_read_message_connection_closed():
    import pytest

    with pytest.raises(ConnectionError):
        read_message(_make_recv(b""))


def test_read_message_partial_header():
    import pytest

    with pytest.raises(ConnectionError):
        read_message(_make_recv(b"\x00\x00"))


def test_multiple_messages():
    raw = pack_message(CMD_START, b"\x03") + pack_message(CMD_SHUTDOWN)
    stream = io.BytesIO(raw)
    recv = stream.read

    t1, p1 = read_message(recv)
    assert t1 == CMD_START
    assert p1 == b"\x03"

    t2, p2 = read_message(recv)
    assert t2 == CMD_SHUTDOWN
    assert p2 == b""


def test_frame_metadata_size():
    assert struct.calcsize(FRAME_META_FMT) == FRAME_META_SIZE


def test_read_message_rejects_oversized():
    import pytest

    # Craft a header claiming a payload larger than MAX_MESSAGE_SIZE
    huge_length = MAX_MESSAGE_SIZE + 2  # +1 for type byte, +1 to exceed
    header = struct.pack(HEADER_FMT, huge_length, MSG_FRAME)
    with pytest.raises(ValueError, match="too large"):
        read_message(_make_recv(header))


# --- CMD_CENSOR round-trip tests ---


def test_censor_header_size():
    assert struct.calcsize(CENSOR_HEADER_FMT) == CENSOR_HEADER_SIZE


def test_censor_box_size():
    assert struct.calcsize(CENSOR_BOX_FMT) == CENSOR_BOX_SIZE


def test_censor_round_trip():
    boxes = [(10, 20, 100, 200), (300, 400, 50, 60)]
    raw = pack_cmd_censor(1, 1920, 1080, boxes)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_CENSOR
    display_id, fw, fh, parsed_boxes = parse_censor_payload(payload)
    assert display_id == 1
    assert fw == 1920
    assert fh == 1080
    assert parsed_boxes == boxes


def test_censor_zero_boxes():
    raw = pack_cmd_censor(2, 3840, 2160, [])
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_CENSOR
    display_id, fw, fh, parsed_boxes = parse_censor_payload(payload)
    assert display_id == 2
    assert fw == 3840
    assert fh == 2160
    assert parsed_boxes == []


def test_censor_many_boxes():
    boxes = [(i * 10, i * 20, i * 5, i * 8) for i in range(100)]
    raw = pack_cmd_censor(1, 7680, 4320, boxes)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_CENSOR
    _, _, _, parsed_boxes = parse_censor_payload(payload)
    assert parsed_boxes == boxes


def test_censor_negative_coords():
    """Signed int32 should preserve negative values."""
    boxes = [(-10, -20, 100, 200)]
    raw = pack_cmd_censor(1, 1920, 1080, boxes)
    _, payload = read_message(_make_recv(raw))
    _, _, _, parsed_boxes = parse_censor_payload(payload)
    assert parsed_boxes == boxes


def test_parse_censor_payload_too_short():
    import pytest

    with pytest.raises(ValueError, match="too short"):
        parse_censor_payload(b"short")


# --- CMD_CENSOR_STYLE tests ---


def test_censor_style_round_trip_blur_and_text():
    raw = pack_cmd_censor_style(blur=1.0, text="BLOCKED")
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_CENSOR_STYLE
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 1.0
    assert pixels == 0.0
    assert text == "BLOCKED"


def test_censor_style_round_trip_no_blur_no_text():
    raw = pack_cmd_censor_style(blur=0.0, text=None)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_CENSOR_STYLE
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 0.0
    assert pixels == 0.0
    assert text is None


def test_censor_style_round_trip_blur_only():
    raw = pack_cmd_censor_style(blur=1.0, text=None)
    _, payload = read_message(_make_recv(raw))
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 1.0
    assert pixels == 0.0
    assert text is None


def test_censor_style_round_trip_text_only():
    raw = pack_cmd_censor_style(blur=0.0, text="NSFW")
    _, payload = read_message(_make_recv(raw))
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 0.0
    assert pixels == 0.0
    assert text == "NSFW"


def test_censor_style_partial_blur():
    raw = pack_cmd_censor_style(blur=0.3, text=None)
    _, payload = read_message(_make_recv(raw))
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 0.30
    assert pixels == 0.0
    assert text is None


def test_censor_style_unicode_text():
    raw = pack_cmd_censor_style(blur=1.0, text="🚫禁止")
    _, payload = read_message(_make_recv(raw))
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 1.0
    assert pixels == 0.0
    assert text == "🚫禁止"


def test_censor_style_pixels_only():
    raw = pack_cmd_censor_style(blur=0.0, pixels=0.8, text=None)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_CENSOR_STYLE
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 0.0
    assert pixels == 0.8
    assert text is None


def test_censor_style_pixels_with_text():
    raw = pack_cmd_censor_style(blur=0.0, pixels=1.0, text="CENSORED")
    _, payload = read_message(_make_recv(raw))
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 0.0
    assert pixels == 1.0
    assert text == "CENSORED"


def test_censor_style_high_intensity():
    """Values >1.0 should round-trip correctly up to 3.0."""
    raw = pack_cmd_censor_style(blur=2.5, pixels=3.0, text=None)
    _, payload = read_message(_make_recv(raw))
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 2.5
    assert pixels == 3.0
    assert text is None


def test_censor_style_clamps_above_max():
    """Values >3.0 should be clamped to 3.0."""
    raw = pack_cmd_censor_style(blur=5.0, pixels=10.0, text=None)
    _, payload = read_message(_make_recv(raw))
    blur, pixels, text = parse_censor_style_payload(payload)
    assert blur == 3.0
    assert pixels == 3.0
    assert text is None


def test_parse_censor_style_payload_too_short():
    import pytest

    with pytest.raises(ValueError, match="too short"):
        parse_censor_style_payload(b"\x01")


# --- Live v2 protocol tests ---


def test_max_message_size_is_256mib():
    assert MAX_MESSAGE_SIZE == 256 * 1024 * 1024


def test_raw_frame_meta_size():
    assert struct.calcsize(RAW_FRAME_META_FMT) == RAW_FRAME_META_SIZE == 24


def test_display_info_size():
    assert struct.calcsize(DISPLAY_INFO_FMT) == DISPLAY_INFO_SIZE == 20


def test_censor_seq_header_size():
    assert struct.calcsize(CENSOR_SEQ_HEADER_FMT) == CENSOR_SEQ_HEADER_SIZE == 18


def test_cmd_start_exact_layout():
    raw = pack_cmd_start(60, 150, 4, 100, True)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_START
    assert payload == struct.pack("!BHBBB", 60, 150, 4, 100, 1)
    assert parse_cmd_start_payload(payload) == (60, 150, 4, 100, True)


def test_cmd_start_defaults():
    raw = pack_cmd_start(45)
    _, payload = read_message(_make_recv(raw))
    assert payload == struct.pack("!BHBBB", 45, 100, 8, 50, 0)


def test_cmd_start_legacy_parse():
    assert parse_cmd_start_payload(b"\x07") == (7, 100, 8, 50, False)


def test_cmd_start_validation():
    import pytest

    with pytest.raises(ValueError):
        pack_cmd_start(0)
    with pytest.raises(ValueError):
        pack_cmd_start(256)
    with pytest.raises(ValueError):
        pack_cmd_start(60, 99, 8, 50, False)
    with pytest.raises(ValueError):
        pack_cmd_start(60, 201, 8, 50, False)
    with pytest.raises(ValueError):
        pack_cmd_start(60, 100, 0, 50, False)
    with pytest.raises(ValueError):
        pack_cmd_start(60, 100, 256, 50, False)
    with pytest.raises(ValueError):
        pack_cmd_start(60, 100, 8, 101, False)
    with pytest.raises(ValueError):
        parse_cmd_start_payload(b"\x01\x02")


def test_raw_frame_round_trip():
    pixels = bytes(range(64))  # 4x4 BGRA
    meta = RawFrameMeta(display_id=2, width=4, height=4, pts_ns=999, seq=41)
    raw = pack_raw_frame(meta, pixels)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == MSG_FRAME_RAW
    parsed_meta, view = parse_raw_frame_payload(payload)
    assert parsed_meta == meta
    assert bytes(view) == pixels


def test_raw_frame_zero_copy():
    header = struct.pack(RAW_FRAME_META_FMT, 1, 2, 1, 0, 7)
    buf = bytearray(header + b"\x01\x02\x03\x04\x05\x06\x07\x08")
    meta, view = parse_raw_frame_payload(buf)
    assert meta == RawFrameMeta(1, 2, 1, 0, 7)
    assert isinstance(view, memoryview)
    # Mutating the source is visible through the view: shared memory, no copy.
    buf[RAW_FRAME_META_SIZE] = 0xFF
    assert view[0] == 0xFF


def test_raw_frame_length_mismatch():
    import pytest

    header = struct.pack(RAW_FRAME_META_FMT, 1, 4, 4, 0, 1)
    with pytest.raises(ValueError):
        parse_raw_frame_payload(header + b"\x00" * 10)
    with pytest.raises(ValueError):
        parse_raw_frame_payload(b"short")


def test_raw_frame_zero_dimension():
    import pytest

    header = struct.pack(RAW_FRAME_META_FMT, 1, 0, 4, 0, 1)
    with pytest.raises(ValueError):
        parse_raw_frame_payload(header)


def test_display_info_round_trip():
    info = DisplayInfo(3, 2560, 1664, 1728, 1117)
    raw = pack_display_info(info)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == MSG_DISPLAY_INFO
    assert parse_display_info_payload(payload) == info


def test_display_info_too_short():
    import pytest

    with pytest.raises(ValueError, match="too short"):
        parse_display_info_payload(b"\x00" * 5)


def test_request_frame_round_trip():
    raw = pack_cmd_request_frame(9)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_REQUEST_FRAME
    assert struct.unpack("!I", payload)[0] == 9


def test_censor_seq_round_trip():
    boxes = [(1, 2, 30, 40), (100, 200, 50, 60)]
    raw = pack_cmd_censor_seq(5, 1728, 1117, 123, boxes)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_CENSOR_SEQ
    assert parse_censor_seq_payload(payload) == (5, 1728, 1117, 123, boxes)


def test_censor_seq_zero_boxes():
    raw = pack_cmd_censor_seq(5, 1728, 1117, 124, [])
    _, payload = read_message(_make_recv(raw))
    assert parse_censor_seq_payload(payload) == (5, 1728, 1117, 124, [])


def test_censor_seq_too_many_boxes():
    import pytest

    with pytest.raises(ValueError):
        pack_cmd_censor_seq(1, 2, 3, 4, [(0, 0, 1, 1)] * 65536)


def test_censor_seq_too_short():
    import pytest

    with pytest.raises(ValueError, match="too short"):
        parse_censor_seq_payload(b"short")
    header = struct.pack(CENSOR_SEQ_HEADER_FMT, 1, 2, 3, 4, 2)
    with pytest.raises(ValueError, match="too short"):
        parse_censor_seq_payload(header + struct.pack("!iiii", 1, 2, 3, 4))


def test_stats_message_round_trip():
    raw = pack_message(MSG_STATS, "fps=60".encode())
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == MSG_STATS
    assert payload.decode() == "fps=60"


def test_read_message_into_single_bytes():
    raw = pack_message(CMD_STOP) + pack_cmd_start(9, 120, 3, 25, False)
    recv_into = _make_recv_into(raw, chunk=1)
    t1, p1 = read_message_into(recv_into)
    assert t1 == CMD_STOP
    assert p1 == bytearray()
    t2, p2 = read_message_into(recv_into)
    assert t2 == CMD_START
    assert bytes(p2) == struct.pack("!BHBBB", 9, 120, 3, 25, 0)


def test_read_message_into_chunked_payload():
    pixels = bytes((i * 7) & 0xFF for i in range(4 * 3 * 4))
    raw = pack_raw_frame(RawFrameMeta(1, 4, 3, 5, 6), pixels)
    recv_into = _make_recv_into(raw, chunk=7)
    msg_type, payload = read_message_into(recv_into)
    assert msg_type == MSG_FRAME_RAW
    assert isinstance(payload, bytearray)
    meta, view = parse_raw_frame_payload(payload)
    assert meta == RawFrameMeta(1, 4, 3, 5, 6)
    assert bytes(view) == pixels


def test_read_message_into_rejects_oversized():
    import pytest

    huge_length = MAX_MESSAGE_SIZE + 2
    header = struct.pack(HEADER_FMT, huge_length, MSG_FRAME_RAW)
    with pytest.raises(ValueError, match="too large"):
        read_message_into(_make_recv_into(header))


def test_read_message_into_rejects_zero_length():
    import pytest

    header = struct.pack(HEADER_FMT, 0, MSG_FRAME_RAW)
    with pytest.raises(ValueError):
        read_message_into(_make_recv_into(header))


def test_read_message_into_closed():
    import pytest

    with pytest.raises(ConnectionError):
        read_message_into(_make_recv_into(b""))
    with pytest.raises(ConnectionError):
        read_message_into(_make_recv_into(b"\x00\x00"))
