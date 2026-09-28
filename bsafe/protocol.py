"""IPC message format: constants, framing, parsing."""

import struct
from dataclasses import dataclass

# Message types: Swift → Python
MSG_FRAME = 0x01
MSG_FRAME_RAW = 0x02
MSG_DISPLAY_INFO = 0x03
MSG_STATS = 0x04

# Message types: Python → Swift
CMD_START = 0x10
CMD_STOP = 0x11
CMD_REQUEST_FRAME = 0x12
CMD_CENSOR = 0x20
CMD_CENSOR_STYLE = 0x21
CMD_CENSOR_SEQ = 0x22
CMD_SHUTDOWN = 0xFF

# Header: 4 bytes length + 1 byte type
HEADER_SIZE = 5
HEADER_FMT = "!IB"  # network byte-order: uint32 length, uint8 type

# Max message payload size (256 MiB — raw BGRA frames are large)
MAX_MESSAGE_SIZE = 256 * 1024 * 1024

# Frame payload header (legacy JPEG): display_id(4) + width(4) + height(4) + timestamp_ns(8)
FRAME_META_SIZE = 20
FRAME_META_FMT = "!IIIq"

# Raw frame payload header: display_id u32 + width u32 + height u32 + pts_ns i64 + seq u32
RAW_FRAME_META_FMT = "!IIIqI"
RAW_FRAME_META_SIZE = struct.calcsize(RAW_FRAME_META_FMT)

# Display info payload: display_id + capture_width + capture_height + points_width + points_height
DISPLAY_INFO_FMT = "!IIIII"
DISPLAY_INFO_SIZE = struct.calcsize(DISPLAY_INFO_FMT)

# Censor payload (legacy): display_id(4) + frame_width(4) + frame_height(4) + box_count(2)
CENSOR_HEADER_FMT = "!IIIH"
CENSOR_HEADER_SIZE = struct.calcsize(CENSOR_HEADER_FMT)
# Per box: x(4) + y(4) + width(4) + height(4) = 16 bytes (signed for safety)
CENSOR_BOX_FMT = "!iiii"
CENSOR_BOX_SIZE = struct.calcsize(CENSOR_BOX_FMT)

# Censor-seq payload header: display_id + frame_width + frame_height + seq + box_count
CENSOR_SEQ_HEADER_FMT = "!IIIIH"
CENSOR_SEQ_HEADER_SIZE = struct.calcsize(CENSOR_SEQ_HEADER_FMT)

# CMD_START payload: fps u8 + scale_percent u16 + persist_passes u8 + smooth_percent u8 + flags u8
START_FMT = "!BHBBB"
START_SIZE = struct.calcsize(START_FMT)


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    display_id: int
    width: int
    height: int
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class RawFrameMeta:
    display_id: int
    width: int
    height: int
    pts_ns: int
    seq: int


@dataclass(frozen=True, slots=True)
class DisplayInfo:
    display_id: int
    capture_width: int
    capture_height: int
    points_width: int
    points_height: int


def pack_message(msg_type: int, payload: bytes = b"") -> bytes:
    """Pack a length-prefixed message: [4B length][1B type][payload]."""
    length = 1 + len(payload)  # type byte + payload
    return struct.pack(HEADER_FMT, length, msg_type) + payload


def read_message(recv_fn) -> tuple[int, bytes]:
    """Read one message from a stream.

    recv_fn(n) should return up to n bytes (like socket.recv).
    Returns (msg_type, payload).
    Raises ConnectionError if the stream closes mid-message.
    Raises ValueError if the message exceeds MAX_MESSAGE_SIZE.
    """
    header = _recv_exact(recv_fn, HEADER_SIZE)
    length, msg_type = struct.unpack(HEADER_FMT, header)
    payload_size = length - 1
    if payload_size > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Message too large: {payload_size} bytes exceeds {MAX_MESSAGE_SIZE} byte limit"
        )
    payload = _recv_exact(recv_fn, payload_size) if payload_size > 0 else b""
    return msg_type, payload


def read_message_into(recv_into) -> tuple[int, bytearray]:
    """Read one message, payload into a single preallocated buffer.

    recv_into(view, nbytes) follows socket.recv_into semantics: it fills
    ``view`` with up to ``nbytes`` bytes and returns the count (0 = closed).
    Returns (msg_type, payload) where payload is ONE preallocated bytearray
    of the exact size (no chunk concatenation).
    Raises ConnectionError if the stream closes mid-message.
    Raises ValueError if the length is 0 or exceeds MAX_MESSAGE_SIZE.
    """
    header = bytearray(HEADER_SIZE)
    _recv_into_exact(recv_into, header)
    length, msg_type = struct.unpack(HEADER_FMT, header)
    if length == 0:
        raise ValueError("Message length 0 is invalid (must include the type byte)")
    payload_size = length - 1
    if payload_size > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Message too large: {payload_size} bytes exceeds {MAX_MESSAGE_SIZE} byte limit"
        )
    payload = bytearray(payload_size)
    if payload_size > 0:
        _recv_into_exact(recv_into, payload)
    return msg_type, payload


def parse_frame_payload(payload: bytes) -> tuple[FrameMetadata, bytes]:
    """Parse a FRAME message payload into metadata + JPEG data."""
    if len(payload) < FRAME_META_SIZE:
        raise ValueError(f"Frame payload too short: {len(payload)} < {FRAME_META_SIZE}")
    display_id, width, height, timestamp_ns = struct.unpack(
        FRAME_META_FMT, payload[:FRAME_META_SIZE]
    )
    jpeg_data = payload[FRAME_META_SIZE:]
    return FrameMetadata(display_id, width, height, timestamp_ns), jpeg_data


def parse_raw_frame_payload(
    payload: bytes | bytearray | memoryview,
) -> tuple[RawFrameMeta, memoryview]:
    """Parse a MSG_FRAME_RAW payload into metadata + zero-copy BGRA view.

    Returns (meta, pixels) where pixels is ``memoryview(payload)[24:]``
    sharing memory with the input (no copy). Rows are tightly packed BGRA.
    """
    if len(payload) < RAW_FRAME_META_SIZE:
        raise ValueError(f"Raw frame payload too short: {len(payload)} < {RAW_FRAME_META_SIZE}")
    display_id, width, height, pts_ns, seq = struct.unpack(
        RAW_FRAME_META_FMT, payload[:RAW_FRAME_META_SIZE]
    )
    if width == 0 or height == 0:
        raise ValueError(f"Raw frame has zero dimension: {width}x{height}")
    pixels = memoryview(payload)[RAW_FRAME_META_SIZE:]
    if len(pixels) != width * height * 4:
        raise ValueError(f"Raw frame pixel size mismatch: {len(pixels)} != {width}x{height}x4")
    return RawFrameMeta(display_id, width, height, pts_ns, seq), pixels


def pack_raw_frame(meta: RawFrameMeta, pixels: bytes) -> bytes:
    """Pack a full framed MSG_FRAME_RAW message (for tests/fixtures)."""
    header = struct.pack(
        RAW_FRAME_META_FMT, meta.display_id, meta.width, meta.height, meta.pts_ns, meta.seq
    )
    return pack_message(MSG_FRAME_RAW, header + bytes(pixels))


def parse_display_info_payload(payload: bytes | bytearray | memoryview) -> DisplayInfo:
    """Parse a MSG_DISPLAY_INFO payload into a DisplayInfo."""
    if len(payload) < DISPLAY_INFO_SIZE:
        raise ValueError(f"Display info payload too short: {len(payload)} < {DISPLAY_INFO_SIZE}")
    display_id, cap_w, cap_h, pts_w, pts_h = struct.unpack(
        DISPLAY_INFO_FMT, payload[:DISPLAY_INFO_SIZE]
    )
    return DisplayInfo(display_id, cap_w, cap_h, pts_w, pts_h)


def pack_display_info(info: DisplayInfo) -> bytes:
    """Pack a full framed MSG_DISPLAY_INFO message."""
    payload = struct.pack(
        DISPLAY_INFO_FMT,
        info.display_id,
        info.capture_width,
        info.capture_height,
        info.points_width,
        info.points_height,
    )
    return pack_message(MSG_DISPLAY_INFO, payload)


def pack_cmd_request_frame(display_id: int) -> bytes:
    """Pack a framed CMD_REQUEST_FRAME credit grant for one display."""
    return pack_message(CMD_REQUEST_FRAME, struct.pack("!I", display_id))


def pack_cmd_censor(
    display_id: int,
    frame_width: int,
    frame_height: int,
    boxes: list[tuple[int, int, int, int]],
) -> bytes:
    """Pack a CMD_CENSOR message with overlay box coordinates."""
    if len(boxes) > 65535:
        raise ValueError(f"Too many boxes: {len(boxes)} exceeds max 65535 (uint16)")
    parts = [struct.pack(CENSOR_HEADER_FMT, display_id, frame_width, frame_height, len(boxes))]
    for x, y, w, h in boxes:
        parts.append(struct.pack(CENSOR_BOX_FMT, x, y, w, h))
    return pack_message(CMD_CENSOR, b"".join(parts))


def parse_censor_payload(
    payload: bytes,
) -> tuple[int, int, int, list[tuple[int, int, int, int]]]:
    """Parse a CMD_CENSOR payload. Returns (display_id, frame_width, frame_height, boxes)."""
    if len(payload) < CENSOR_HEADER_SIZE:
        raise ValueError(f"Censor payload too short: {len(payload)} < {CENSOR_HEADER_SIZE}")
    display_id, frame_width, frame_height, box_count = struct.unpack(
        CENSOR_HEADER_FMT, payload[:CENSOR_HEADER_SIZE]
    )
    expected = CENSOR_HEADER_SIZE + box_count * CENSOR_BOX_SIZE
    if len(payload) < expected:
        raise ValueError(
            f"Censor payload too short for {box_count} boxes: {len(payload)} < {expected}"
        )
    boxes = []
    offset = CENSOR_HEADER_SIZE
    for _ in range(box_count):
        x, y, w, h = struct.unpack(CENSOR_BOX_FMT, payload[offset : offset + CENSOR_BOX_SIZE])
        boxes.append((x, y, w, h))
        offset += CENSOR_BOX_SIZE
    return display_id, frame_width, frame_height, boxes


def pack_cmd_censor_seq(
    display_id: int,
    frame_width: int,
    frame_height: int,
    seq: int,
    boxes: list[tuple[int, int, int, int]],
) -> bytes:
    """Pack a CMD_CENSOR_SEQ message: boxes in capture pixels of frame ``seq``."""
    if len(boxes) > 65535:
        raise ValueError(f"Too many boxes: {len(boxes)} exceeds max 65535 (uint16)")
    parts = [
        struct.pack(CENSOR_SEQ_HEADER_FMT, display_id, frame_width, frame_height, seq, len(boxes))
    ]
    for x, y, w, h in boxes:
        parts.append(struct.pack(CENSOR_BOX_FMT, x, y, w, h))
    return pack_message(CMD_CENSOR_SEQ, b"".join(parts))


def parse_censor_seq_payload(
    payload: bytes | bytearray | memoryview,
) -> tuple[int, int, int, int, list[tuple[int, int, int, int]]]:
    """Parse a CMD_CENSOR_SEQ payload. Returns (display_id, frame_width, frame_height, seq, boxes)."""
    if len(payload) < CENSOR_SEQ_HEADER_SIZE:
        raise ValueError(f"Censor-seq payload too short: {len(payload)} < {CENSOR_SEQ_HEADER_SIZE}")
    display_id, frame_width, frame_height, seq, box_count = struct.unpack(
        CENSOR_SEQ_HEADER_FMT, payload[:CENSOR_SEQ_HEADER_SIZE]
    )
    expected = CENSOR_SEQ_HEADER_SIZE + box_count * CENSOR_BOX_SIZE
    if len(payload) < expected:
        raise ValueError(
            f"Censor-seq payload too short for {box_count} boxes: {len(payload)} < {expected}"
        )
    boxes = []
    offset = CENSOR_SEQ_HEADER_SIZE
    for _ in range(box_count):
        x, y, w, h = struct.unpack(CENSOR_BOX_FMT, payload[offset : offset + CENSOR_BOX_SIZE])
        boxes.append((x, y, w, h))
        offset += CENSOR_BOX_SIZE
    return display_id, frame_width, frame_height, seq, boxes


def pack_cmd_censor_style(blur: float, pixels: float = 0.0, text: str | None = None) -> bytes:
    """Pack a CMD_CENSOR_STYLE message.

    Payload: [2B blur_hundredths][2B pixels_hundredths][2B text_length][text_bytes (UTF-8)]
    Values are hundredths: 100 = 1.0 (100%), 300 = 3.0x multiplier (max 3.0).
    text_length=0 means no text.
    """
    blur_pct = max(0, min(300, round(blur * 100)))
    pixels_pct = max(0, min(300, round(pixels * 100)))
    text_bytes = text.encode("utf-8") if text else b""
    payload = struct.pack("!HHH", blur_pct, pixels_pct, len(text_bytes)) + text_bytes
    return pack_message(CMD_CENSOR_STYLE, payload)


def parse_censor_style_payload(payload: bytes) -> tuple[float, float, str | None]:
    """Parse a CMD_CENSOR_STYLE payload. Returns (blur, pixels, text).

    blur is a float 0.0–3.0 (1.0 = 100% blur, >1 multiplies effect).
    pixels is a float 0.0–3.0 (1.0 = 100% pixelation, >1 multiplies effect).
    """
    if len(payload) < 6:
        raise ValueError(f"Censor style payload too short: {len(payload)} < 6")
    blur_pct, pixels_pct, text_length = struct.unpack("!HHH", payload[:6])
    if len(payload) < 6 + text_length:
        raise ValueError(
            f"Censor style payload too short for text: {len(payload)} < {6 + text_length}"
        )
    text = payload[6 : 6 + text_length].decode("utf-8") if text_length > 0 else None
    return blur_pct / 100.0, pixels_pct / 100.0, text


def pack_cmd_start(
    fps: int,
    scale_percent: int = 100,
    persist_passes: int = 8,
    smooth_percent: int = 50,
    stats: bool = False,
) -> bytes:
    """Pack a CMD_START message (6-byte payload: fps, scale, persist, smooth, flags)."""
    if not 1 <= fps <= 255:
        raise ValueError(f"fps must be in 1..255, got {fps}")
    if not 100 <= scale_percent <= 200:
        raise ValueError(f"scale_percent must be in 100..200, got {scale_percent}")
    if not 1 <= persist_passes <= 255:
        raise ValueError(f"persist_passes must be in 1..255, got {persist_passes}")
    if not 0 <= smooth_percent <= 100:
        raise ValueError(f"smooth_percent must be in 0..100, got {smooth_percent}")
    payload = struct.pack(
        START_FMT, fps, scale_percent, persist_passes, smooth_percent, 1 if stats else 0
    )
    return pack_message(CMD_START, payload)


def parse_cmd_start_payload(
    payload: bytes | bytearray | memoryview,
) -> tuple[int, int, int, int, bool]:
    """Parse a CMD_START payload. Returns (fps, scale_percent, persist_passes, smooth_percent, stats).

    Accepts the 6-byte payload and the legacy 1-byte payload (fps only;
    then scale 100, persist 8, smooth 50, flags 0).
    """
    if len(payload) == 1:
        return (payload[0], 100, 8, 50, False)
    if len(payload) == START_SIZE:
        fps, scale_percent, persist_passes, smooth_percent, flags = struct.unpack(
            START_FMT, payload[:START_SIZE]
        )
        return (fps, scale_percent, persist_passes, smooth_percent, bool(flags & 0x01))
    raise ValueError(f"CMD_START payload must be 1 or {START_SIZE} bytes, got {len(payload)}")


def _recv_exact(recv_fn, n: int) -> bytes:
    """Read exactly n bytes from recv_fn."""
    buf = bytearray()
    while len(buf) < n:
        chunk = recv_fn(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading message")
        buf.extend(chunk)
    return bytes(buf)


def _recv_into_exact(recv_into, buf: bytearray) -> None:
    """Fill ``buf`` exactly using recv_into(view, nbytes) semantics."""
    view = memoryview(buf)
    offset = 0
    total = len(buf)
    while offset < total:
        n = recv_into(view[offset:], total - offset)
        if n == 0:
            raise ConnectionError("Connection closed while reading message")
        offset += n
