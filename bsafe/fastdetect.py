"""Full-frame NudeNet detection at native resolution (live v2).

Runs NudeNet YOLO on the whole frame at its native size (no letterbox, no
resize) through ONNX Runtime with static shapes (CoreML EP where available).
Frames stay in memory; no temp files are used.
"""

import logging
import os
import time

import cv2
import numpy as np

from bsafe.detector import Detection

logger = logging.getLogger(__name__)

_EXTRA_WARN_INTERVAL_S = 5.0
_last_extra_warn_s = float("-inf")

# Class labels in NudeNet 3.4.2 order (nudenet.nudenet.__labels).
NUDENET_LABELS: list[str] = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

# YOLO output layout: rows 0..3 are cx, cy, w, h; rows 4..21 are class scores.
_NUM_CLASSES = len(NUDENET_LABELS)
_SCORE_OFFSET = 4
_DETECT_THRESHOLD = 0.2
_NMS_SCORE = 0.25
_NMS_IOU = 0.45


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Convert a BGR/BGRA frame to a NudeNet RGB blob.

    Returns (blob, W32, H32) where blob is float32 (1, 3, H32, W32) in RGB
    order scaled by 1/255, zero-padded on the bottom/right up to multiples
    of 32. No Python loops over pixels.
    """
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 frame, got shape {frame.shape}")
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Frame has zero dimension: {w}x{h}")
    w32 = -(-w // 32) * 32
    h32 = -(-h // 32) * 32
    if frame.shape[2] == 4:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if h32 != h or w32 != w:
        rgb = cv2.copyMakeBorder(rgb, 0, h32 - h, 0, w32 - w, cv2.BORDER_CONSTANT, value=0)
    blob = cv2.dnn.blobFromImage(rgb, 1 / 255.0, (w32, h32), (0, 0, 0), swapRB=False, crop=False)
    return blob, w32, h32


def postprocess(
    output: np.ndarray,
    frame_width: int,
    frame_height: int,
    min_confidence: float = 0.0,
) -> list[Detection]:
    """Decode a NudeNet output0 tensor into detections.

    ``output`` has shape (1, 22, N). Vectorized over N (no Python loop over
    rows); semantics match nudenet 3.4.2 ``_postprocess`` with no
    letterbox/scaling (model input pixels == frame pixels). Returned order
    follows class-agnostic NMS order.
    """
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != _SCORE_OFFSET + _NUM_CLASSES:
        raise ValueError(f"Expected output shape (1, 22, N), got {output.shape}")
    out = output[0]  # (22, N)
    scores = out[_SCORE_OFFSET : _SCORE_OFFSET + _NUM_CLASSES]  # (18, N)
    class_ids = np.argmax(scores, axis=0)  # (N,)
    max_scores = np.max(scores, axis=0)  # (N,)
    keep = max_scores >= _DETECT_THRESHOLD
    if not bool(np.any(keep)):
        return []
    cx = out[0][keep]
    cy = out[1][keep]
    bw = out[2][keep]
    bh = out[3][keep]
    kept_classes = class_ids[keep]
    kept_scores = max_scores[keep]
    x0 = np.clip(cx - bw / 2.0, 0, frame_width)
    x1 = np.clip(cx + bw / 2.0, 0, frame_width)
    y0 = np.clip(cy - bh / 2.0, 0, frame_height)
    y1 = np.clip(cy + bh / 2.0, 0, frame_height)
    w = x1 - x0
    h = y1 - y0
    visible = (w > 0) & (h > 0)
    if not bool(np.any(visible)):
        return []
    x0 = x0[visible]
    y0 = y0[visible]
    w = w[visible]
    h = h[visible]
    kept_classes = kept_classes[visible]
    kept_scores = kept_scores[visible]
    boxes = np.stack([x0, y0, w, h], axis=1).tolist()
    scores_list = kept_scores.tolist()
    indices = cv2.dnn.NMSBoxes(boxes, scores_list, _NMS_SCORE, _NMS_IOU)
    if len(indices) == 0:
        return []
    order = np.asarray(indices).reshape(-1)
    detections: list[Detection] = []
    for flat in order:
        j = int(flat)
        confidence = float(kept_scores[j])
        if confidence < min_confidence:
            continue
        bx, by = int(x0[j]), int(y0[j])
        bw_int, bh_int = int(w[j]), int(h[j])
        if bw_int <= 0 or bh_int <= 0:
            continue
        detections.append(
            Detection(
                class_name=NUDENET_LABELS[int(kept_classes[j])],
                confidence=confidence,
                box=(bx, by, bw_int, bh_int),
            )
        )
    return detections


class FullFrameNudeDetector:
    """NudeNet YOLO on the whole frame at its native resolution (no letterbox, no resize)."""

    def __init__(
        self,
        model_path: str | None = None,
        min_confidence: float = 0.0,
        *,
        use_coreml: bool = True,
        compute_units: str = "CPUAndGPU",
    ):
        import onnxruntime

        onnxruntime.set_default_logger_severity(3)
        if model_path is None:
            import nudenet

            model_path = os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.use_coreml = use_coreml
        self.compute_units = compute_units
        self._sessions: dict[tuple[int, int], tuple] = {}
        self._session = None
        self._input_name: str | None = None
        self._current_key: tuple[int, int] | None = None
        self._provider = "unprepared"
        self._coreml_warned = False

    @property
    def provider(self) -> str:
        """Provider used by the most recent session: 'coreml', 'cpu', or 'unprepared'."""
        return self._provider

    def prepare(self, width: int, height: int) -> None:
        """Create (or reuse) a static-shape session for this frame size.

        Idempotent per padded shape; padded dims are ceil(width/32)*32 and
        ceil(height/32)*32. Warms up with one run on zeros.
        """
        import onnxruntime

        if width <= 0 or height <= 0:
            raise ValueError(f"Frame has zero dimension: {width}x{height}")
        w32 = -(-int(width) // 32) * 32
        h32 = -(-int(height) // 32) * 32
        key = (h32, w32)
        if key == self._current_key and self._session is not None:
            return
        if key in self._sessions:
            session, input_name, provider = self._sessions[key]
            self._session = session
            self._input_name = input_name
            self._current_key = key
            self._provider = provider
            return
        options = onnxruntime.SessionOptions()
        options.add_free_dimension_override_by_name("batch", 1)
        options.add_free_dimension_override_by_name("height", h32)
        options.add_free_dimension_override_by_name("width", w32)
        session = None
        input_name = None
        provider = "cpu"
        if self.use_coreml and "CoreMLExecutionProvider" in onnxruntime.get_available_providers():
            try:
                coreml_session = onnxruntime.InferenceSession(
                    self.model_path,
                    sess_options=options,
                    providers=[
                        (
                            "CoreMLExecutionProvider",
                            {
                                "ModelFormat": "MLProgram",
                                "MLComputeUnits": self.compute_units,
                            },
                        ),
                        "CPUExecutionProvider",
                    ],
                )
                coreml_input = coreml_session.get_inputs()[0].name
                coreml_session.run(
                    None, {coreml_input: np.zeros((1, 3, h32, w32), dtype=np.float32)}
                )
                session = coreml_session
                input_name = coreml_input
                provider = (
                    "coreml" if "CoreMLExecutionProvider" in session.get_providers() else "cpu"
                )
            except Exception:
                if not self._coreml_warned:
                    logger.warning("CoreML session failed, falling back to CPU")
                    self._coreml_warned = True
                session = None
                input_name = None
        if session is None:
            cpu_session = onnxruntime.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            cpu_input = cpu_session.get_inputs()[0].name
            cpu_session.run(None, {cpu_input: np.zeros((1, 3, h32, w32), dtype=np.float32)})
            session = cpu_session
            input_name = cpu_input
            provider = "cpu"
        self._sessions[key] = (session, input_name, provider)
        self._session = session
        self._input_name = input_name
        self._current_key = key
        self._provider = provider

    def detect_bgra(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on an HxWx4 BGRA (or HxWx3 BGR) uint8 frame.

        The frame may be a non-owning view over a memoryview. Calls
        prepare() for the frame size when needed.
        """
        height, width = int(frame.shape[0]), int(frame.shape[1])
        self.prepare(width, height)
        blob, _, _ = preprocess(frame)
        assert self._session is not None and self._input_name is not None
        output = self._session.run(None, {self._input_name: blob})[0]
        return postprocess(output, width, height, self.min_confidence)

    def close(self) -> None:
        """Drop cached sessions. Safe to call twice."""
        self._sessions.clear()
        self._session = None
        self._input_name = None
        self._current_key = None


def extra_size(width: int, height: int, factor: float) -> tuple[int, int]:
    """Small-frame size for one extra scale factor (w_small, h_small)."""
    w_small = max(32, round(int(width) * float(factor)))
    h_small = max(32, round(int(height) * float(factor)))
    return (w_small, h_small)


def prepare_multiscale(detector, width: int, height: int, factors=None) -> None:
    """Warm the primary shape plus every extra shape ``detect_multiscale`` uses.

    Calls ``detector.prepare`` for (width, height) and for each
    ``extra_size(width, height, f)``. No-op when the detector has no
    ``prepare`` (e.g. EraX). Extra sizes are computed with the same shared
    helper as :func:`detect_multiscale` so warmup always matches inference.
    """
    if not hasattr(detector, "prepare"):
        return
    detector.prepare(int(width), int(height))
    if not factors:
        return
    for f in factors:
        w_small, h_small = extra_size(width, height, f)
        detector.prepare(w_small, h_small)


def detect_multiscale(detector, frame: np.ndarray, factors) -> list[Detection]:
    """Run the primary full-frame pass plus extra downscaled passes.

    The primary pass is ``detector.detect_bgra(frame)`` unchanged. Then for
    each factor ``f`` in ``factors`` (0 < f < 1, relative to the frame size)
    the frame is downscaled with ``cv2.INTER_AREA`` and run through
    ``detect_bgra`` again; each small-frame box is rescaled back to frame
    coordinates with the exact ratios ``W/w_small`` and ``H/h_small``,
    rounded to int, clamped to the frame bounds (w/h >= 1).

    Returns the concatenation (primary first). No dedup — downstream
    ``build_censor_boxes`` merges overlapping boxes. ``FullFrameNudeDetector``
    already caches one static-shape session per padded shape, so alternating
    shapes just swaps the cached ``_session`` via the dict.

    Each extra pass is isolated: a failure (resize, prepare, session.run)
    logs a rate-limited warning and keeps the primary detections.
    Primary-pass exceptions still propagate.
    """
    detections: list[Detection] = list(detector.detect_bgra(frame))
    if not factors:
        return detections
    h, w = int(frame.shape[0]), int(frame.shape[1])
    for f in factors:
        try:
            w_small, h_small = extra_size(w, h, f)
            small = cv2.resize(frame, (w_small, h_small), interpolation=cv2.INTER_AREA)
            sx = w / w_small
            sy = h / h_small
            for d in detector.detect_bgra(small):
                bx, by, bw, bh = d.box
                x = int(round(bx * sx))
                y = int(round(by * sy))
                nw = int(round(bw * sx))
                nh = int(round(bh * sy))
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                nw = max(1, min(nw, w - x))
                nh = max(1, min(nh, h - y))
                detections.append(
                    Detection(class_name=d.class_name, confidence=d.confidence, box=(x, y, nw, nh))
                )
        except Exception as e:
            global _last_extra_warn_s
            now = time.monotonic()
            if now - _last_extra_warn_s >= _EXTRA_WARN_INTERVAL_S:
                logger.warning("extra detection scale %s failed: %s", f, e)
                _last_extra_warn_s = now
            continue
    return detections
