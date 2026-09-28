"""Live v2 screen censoring loop: raw frames, GPU detection, seq-tagged boxes.

Python grants frame credits (at most one outstanding per display). Swift answers
with the newest raw BGRA frame. Python runs full-frame NudeNet at the frame's
native resolution, replies with boxes tagged with the frame seq, then grants the
next credit. Swift catches boxes up and merges them into its native tracks.
Frames stay in memory only; never written to disk, never logged.
"""

import logging
import queue
import time

import numpy as np

from bsafe.style import dim, info

logger = logging.getLogger(__name__)

ALLOWED_DETECT_SCALES = (1.0, 1.25, 1.5, 2.0)
STATS_INTERVAL_S = 2.0
_ERROR_LOG_INTERVAL_S = 5.0

AUTO_EXTRA_SCALES: tuple[float, ...] = (0.5,)
MAX_EXTRA_SCALES = 3


def parse_extra_scales(raw) -> tuple[float, ...] | None:
    """Parse an --extra-scales value. Returns None for auto (raw is None).

    Accepts a TOML/CLI string (comma-separated floats, "none"/"" disables),
    a single float/int, or a list/tuple of floats. Raises ValueError on
    unparseable input.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return (float(raw),)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return ()
        try:
            return tuple(float(v) for v in raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid --extra-scales {raw!r}: expected floats") from e
    s = str(raw).strip()
    if s == "" or s.lower() == "none":
        return ()
    parts = [p.strip() for p in s.split(",")]
    if any(p == "" for p in parts):
        raise ValueError(f"invalid --extra-scales {raw!r}: expected comma-separated floats")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"invalid --extra-scales {raw!r}: expected comma-separated floats") from e


def validate_extra_scales(
    scales: tuple[float, ...] | list[float] | None, detect_scale: float
) -> None:
    """Validate point-size extra scales against --detect-scale. Raises ValueError."""
    if scales is None:
        return
    scales = tuple(scales)
    if len(scales) > MAX_EXTRA_SCALES:
        raise ValueError(f"invalid --extra-scales {list(scales)}: at most 3 values")
    if len(set(scales)) != len(scales):
        raise ValueError(f"invalid --extra-scales {list(scales)}: duplicate values")
    for v in scales:
        if not (0 < v < detect_scale):
            raise ValueError(
                f"invalid --extra-scales {v}: each scale must satisfy 0 < s < --detect-scale {detect_scale}"
            )


def resolve_extra_scales(model, detect_scale: float, raw) -> tuple[float, ...]:
    """Resolve raw --extra-scales (None = auto) to point-size scales. Raises ValueError."""
    from bsafe.detector import get_model_backend

    parsed = parse_extra_scales(raw)
    if parsed is None:
        backend = get_model_backend(model)
        if backend == "erax":
            return ()
        return tuple(AUTO_EXTRA_SCALES)
    backend = get_model_backend(model)
    if backend == "erax":
        # EraX ignores extra scales (CLI warns); skip validation.
        return ()
    validate_extra_scales(parsed, detect_scale)
    return tuple(parsed)


def validate_live_options(model, detect_scale: float, extra_scales=None) -> None:
    """Reject unsupported detect scales and EraX combinations. Raises ValueError."""
    if detect_scale not in ALLOWED_DETECT_SCALES:
        allowed = ", ".join(str(s) for s in ALLOWED_DETECT_SCALES)
        raise ValueError(f"unsupported --detect-scale {detect_scale}. Allowed values: {allowed}")
    from bsafe.detector import get_model_backend

    backend = get_model_backend(model)
    if backend == "erax" and detect_scale != 1.0:
        raise ValueError("--detect-scale is NudeNet-only; use 1.0 with EraX models")
    if extra_scales is None:
        return
    # Parse for every backend so unparseable input fails before spawn;
    # only a successfully parsed value is ignored for EraX (CLI warns).
    parsed = parse_extra_scales(extra_scales)
    if parsed is None:
        return
    if backend == "erax":
        return
    validate_extra_scales(parsed, detect_scale)


class LiveStats:
    """Window + lifetime detection timing stats (no frames/pixels saved)."""

    def __init__(self, time_fn=time.monotonic, interval_s: float = STATS_INTERVAL_S):
        self._time_fn = time_fn
        self._interval = interval_s
        self._start = time_fn()
        self._win_start = self._start
        self._last_log = self._start
        self._win_n = 0
        self._win_detect = 0.0
        self._win_detect_max = 0.0
        self._win_r2s = 0.0
        self._life_n = 0
        self._life_detect = 0.0
        self._life_detect_max = 0.0
        self._life_r2s = 0.0

    def record(self, detect_s: float, receive_to_send_s: float) -> None:
        d = max(0.0, float(detect_s))
        r = max(0.0, float(receive_to_send_s))
        self._win_n += 1
        self._win_detect += d
        self._win_detect_max = max(self._win_detect_max, d)
        self._win_r2s += r
        self._life_n += 1
        self._life_detect += d
        self._life_detect_max = max(self._life_detect_max, d)
        self._life_r2s += r

    def maybe_log(self) -> str | None:
        now = self._time_fn()
        if now - self._last_log < self._interval:
            return None
        self._last_log = now
        line = self._format(now, window=True)
        self._win_n = 0
        self._win_detect = 0.0
        self._win_detect_max = 0.0
        self._win_r2s = 0.0
        self._win_start = now
        return line

    def summary(self) -> str:
        return self._format(self._time_fn(), window=False)

    def _format(self, now: float, window: bool) -> str:
        if window:
            n = self._win_n
            elapsed = now - self._win_start
            sum_d = self._win_detect
            max_d = self._win_detect_max
            sum_r = self._win_r2s
            scope = "window"
        else:
            n = self._life_n
            elapsed = now - self._start
            sum_d = self._life_detect
            max_d = self._life_detect_max
            sum_r = self._life_r2s
            scope = "total"
        rate = (n / elapsed) if elapsed > 0 and n else 0.0
        avg_detect_ms = (sum_d / n * 1000.0) if n else 0.0
        max_detect_ms = (max_d * 1000.0) if n else 0.0
        avg_r2s_ms = (sum_r / n * 1000.0) if n else 0.0
        return (
            f"live stats ({scope}): frames={n} rate={rate:.1f}/s "
            f"avg_detect_ms={avg_detect_ms:.1f} max_detect_ms={max_detect_ms:.1f} "
            f"avg_receive_to_send_ms={avg_r2s_ms:.1f}"
        )


def bgra_view(meta, pixels) -> np.ndarray:
    """Zero-copy BGRA view over raw frame pixels as HxWx4 uint8."""
    return np.frombuffer(pixels, dtype=np.uint8).reshape(meta.height, meta.width, 4)


class LiveSession:
    """Testable heart of cmd_start: credits, detection, seq-tagged boxes."""

    def __init__(
        self,
        server,
        detector,
        *,
        censor_classes,
        padding: float,
        full_censor: bool,
        is_nudenet: bool,
        stats: LiveStats | None = None,
        print_fn=print,
        time_fn=time.monotonic,
        min_padding: int = 0,
        extra_factors: tuple[float, ...] = (),
    ):
        self._server = server
        self._detector = detector
        self._censor_classes = censor_classes
        self._padding = padding
        self._min_padding = min_padding
        self._full_censor = full_censor
        self._is_nudenet = is_nudenet
        self._stats = stats
        self._print_fn = print_fn
        self._time_fn = time_fn
        self._last_exc_log = float("-inf")
        self._extra_factors = tuple(extra_factors)

    def _emit(self, text: str) -> None:
        try:
            self._print_fn(text, flush=True)
        except TypeError:
            self._print_fn(text)

    def handle_display_info(self, display_info) -> None:
        cw = display_info.capture_width
        ch = display_info.capture_height
        pw = display_info.points_width
        ph = display_info.points_height
        did = display_info.display_id
        self._emit(
            f"{info(f'Display {did}')}: capturing {info(f'{cw}x{ch}')} px {dim(f'({pw}x{ph} pt)')}"
        )
        if hasattr(self._detector, "prepare"):
            try:
                t0 = self._time_fn()
                from bsafe.fastdetect import prepare_multiscale

                prepare_multiscale(self._detector, cw, ch, self._extra_factors)
                elapsed_ms = (self._time_fn() - t0) * 1000.0
                provider = getattr(self._detector, "provider", "cpu")
                self._emit(
                    f"Detector ready for {info(f'{cw}x{ch}')} "
                    f"({provider}, {info(f'{elapsed_ms:.1f} ms')})"
                )
            except Exception:
                now = self._time_fn()
                if now - self._last_exc_log >= _ERROR_LOG_INTERVAL_S:
                    logger.exception("Live detector prepare failed")
                    self._last_exc_log = now
        self._server.request_frame(did)

    def drain_display_events(self) -> int:
        count = 0
        while True:
            try:
                display_info = self._server.display_events.get_nowait()
            except queue.Empty:
                break
            self.handle_display_info(display_info)
            count += 1
        return count

    def step(self, timeout: float = 0.1) -> bool:
        self.drain_display_events()
        try:
            meta, pixels, receipt_mono = self._server.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return False
        detect_start = self._time_fn()
        detect_s = 0.0
        try:
            frame = bgra_view(meta, pixels)
            if self._is_nudenet:
                if self._extra_factors:
                    from bsafe.fastdetect import detect_multiscale

                    detections = detect_multiscale(self._detector, frame, self._extra_factors)
                else:
                    detections = self._detector.detect_bgra(frame)
            else:
                import cv2

                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                detections = self._detector.detect_frame(bgr)
            detect_s = self._time_fn() - detect_start
            logger.debug("Processed frame: %d detection(s)", len(detections))
            if detections:
                for d in detections:
                    logger.debug(
                        "[detection] %s (%.2f) at %s on display %d",
                        d.class_name,
                        d.confidence,
                        d.box,
                        meta.display_id,
                    )
            from bsafe.censor import (
                FULL_CENSOR_MULTIPLIER,
                build_censor_boxes,
                expand_boxes,
            )

            boxes = build_censor_boxes(
                detections,
                self._censor_classes,
                self._padding,
                meta.width,
                meta.height,
                self._min_padding,
            )
            if self._full_censor:
                boxes = expand_boxes(boxes, FULL_CENSOR_MULTIPLIER, meta.width, meta.height)
        except Exception:
            now = self._time_fn()
            if now - self._last_exc_log >= _ERROR_LOG_INTERVAL_S:
                logger.exception("Live detection failed")
                self._last_exc_log = now
            detect_s = now - detect_start
            boxes = []
        self._server.send_censor_seq(meta.display_id, meta.width, meta.height, meta.seq, boxes)
        self._server.request_frame(meta.display_id)
        if self._stats is not None:
            send_mono = self._time_fn()
            self._stats.record(detect_s, send_mono - receipt_mono)
            line = self._stats.maybe_log()
            if line is not None:
                self._emit(dim(line))
        return True
