"""Unit tests for full-frame native-resolution NudeNet detection."""

import logging
import os
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

from bsafe.detector import Detection
from bsafe.fastdetect import (
    NUDENET_LABELS,
    FullFrameNudeDetector,
    detect_multiscale,
    extra_size,
    postprocess,
    prepare_multiscale,
    preprocess,
)

FIXTURE_JPG = os.path.join(os.path.dirname(__file__), "fixtures", "sample_frame.jpg")


def _make_output(specs, width=200, height=200):
    """Build an output0 tensor (1, 22, N) from (cx, cy, w, h, class_idx, score) specs."""
    n = len(specs)
    out = np.zeros((1, 22, n), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(specs):
        out[0, 0, i] = cx
        out[0, 1, i] = cy
        out[0, 2, i] = w
        out[0, 3, i] = h
        out[0, 4 + cls, i] = score
    return out, width, height


def test_labels_match_nudenet():
    import nudenet.nudenet

    assert NUDENET_LABELS == list(getattr(nudenet.nudenet, "__labels"))
    assert len(NUDENET_LABELS) == 18


def test_preprocess_shape_dtype_padding():
    frame = np.zeros((45, 70, 4), dtype=np.uint8)
    blob, w32, h32 = preprocess(frame)
    assert (w32, h32) == (96, 64)
    assert blob.shape == (1, 3, 64, 96)
    assert blob.dtype == np.float32
    # Padding region (bottom rows, right cols) is zeros.
    assert bool((blob[:, :, 45:, :] == 0).all())
    assert bool((blob[:, :, :, 70:] == 0).all())


def test_preprocess_channel_order():
    frame = np.zeros((45, 70, 4), dtype=np.uint8)
    frame[10, 20] = (10, 20, 30, 255)  # BGRA: B=10, G=20, R=30
    blob, _, _ = preprocess(frame)
    # Blob is RGB order scaled by 1/255.
    assert blob[0, 0, 10, 20] == pytest.approx(30 / 255)
    assert blob[0, 1, 10, 20] == pytest.approx(20 / 255)
    assert blob[0, 2, 10, 20] == pytest.approx(10 / 255)


def test_preprocess_bgr_input():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[5, 6] = (7, 8, 9)  # BGR
    blob, w32, h32 = preprocess(frame)
    assert (w32, h32) == (32, 32)
    assert blob.shape == (1, 3, 32, 32)
    assert blob[0, 0, 5, 6] == pytest.approx(9 / 255)
    assert blob[0, 2, 5, 6] == pytest.approx(7 / 255)


def test_postprocess_center_to_corner_and_class():
    out, w, h = _make_output([(50, 60, 20, 30, 3, 0.9)])
    dets = postprocess(out, w, h)
    assert len(dets) == 1
    assert dets[0].class_name == "FEMALE_BREAST_EXPOSED"
    assert dets[0].confidence == pytest.approx(0.9)
    assert dets[0].box == (40, 45, 20, 30)


def test_postprocess_threshold():
    out, w, h = _make_output([(50, 50, 20, 20, 6, 0.19)])
    assert postprocess(out, w, h) == []
    # Scores in [0.2, 0.25) pass the prefilter but NudeNet's NMS score threshold (0.25) drops them.
    out, w, h = _make_output([(50, 50, 20, 20, 6, 0.24)])
    assert postprocess(out, w, h) == []
    out, w, h = _make_output([(50, 50, 20, 20, 6, 0.3)])
    assert len(postprocess(out, w, h)) == 1


def test_postprocess_clipping():
    # Box hanging off the right/bottom edges is clipped into the frame.
    out, w, h = _make_output([(195, 195, 20, 20, 6, 0.9)], width=200, height=200)
    dets = postprocess(out, w, h)
    assert len(dets) == 1
    assert dets[0].box == (185, 185, 15, 15)


def test_postprocess_nms_suppresses_overlap():
    out, w, h = _make_output(
        [
            (50, 50, 20, 20, 3, 0.9),
            (52, 52, 20, 20, 3, 0.7),  # heavy overlap, lower score
            (150, 150, 20, 20, 3, 0.8),  # far away, kept
        ]
    )
    dets = postprocess(out, w, h)
    assert len(dets) == 2
    assert dets[0].confidence == pytest.approx(0.9)
    assert dets[0].box == (40, 40, 20, 20)
    assert dets[1].box == (140, 140, 20, 20)


def test_postprocess_min_confidence_filter():
    out, w, h = _make_output([(50, 50, 20, 20, 6, 0.3)])
    assert len(postprocess(out, w, h, min_confidence=0.0)) == 1
    assert postprocess(out, w, h, min_confidence=0.5) == []


def test_postprocess_argmax_class():
    out = np.zeros((1, 22, 1), dtype=np.float32)
    out[0, 0, 0] = 50
    out[0, 1, 0] = 50
    out[0, 2, 0] = 20
    out[0, 3, 0] = 20
    out[0, 4 + 5, 0] = 0.4
    out[0, 4 + 12, 0] = 0.85
    dets = postprocess(out, 200, 200)
    assert len(dets) == 1
    assert dets[0].class_name == "FACE_MALE"
    assert dets[0].confidence == pytest.approx(0.85)


def test_postprocess_empty():
    out = np.zeros((1, 22, 10), dtype=np.float32)
    assert postprocess(out, 200, 200) == []


def test_end_to_end_cpu_and_prepare_idempotent():
    bgr = cv2.imread(FIXTURE_JPG, cv2.IMREAD_COLOR)
    assert bgr is not None
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    det = FullFrameNudeDetector(use_coreml=False)
    try:
        assert det.provider == "unprepared"
        dets = det.detect_bgra(bgra)
        assert isinstance(dets, list)
        assert all(isinstance(d, Detection) for d in dets)
        assert det.provider == "cpu"
        h, w = bgra.shape[:2]
        det.prepare(w, h)
        det.prepare(w, h)
        assert len(det._sessions) == 1
    finally:
        det.close()
        det.close()


class _WarmupFailSession:
    def get_providers(self):
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, *args, **kwargs):
        raise RuntimeError("coreml warmup boom")


class _WorkingCpuSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, *args, **kwargs):
        return [np.zeros((1, 22, 1), dtype=np.float32)]


class _CpuFailSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, *args, **kwargs):
        raise RuntimeError("cpu warmup boom")


def test_prepare_falls_back_when_coreml_warmup_fails(caplog):
    import onnxruntime

    coreml_fail = _WarmupFailSession()
    cpu_ok = _WorkingCpuSession()
    coreml_fail2 = _WarmupFailSession()
    cpu_ok2 = _WorkingCpuSession()
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    with (
        mock.patch.object(onnxruntime, "get_available_providers", return_value=providers),
        mock.patch.object(
            onnxruntime,
            "InferenceSession",
            side_effect=[coreml_fail, cpu_ok, coreml_fail2, cpu_ok2],
        ),
    ):
        det = FullFrameNudeDetector(model_path="dummy.onnx", use_coreml=True)
        with caplog.at_level(logging.WARNING, logger="bsafe.fastdetect"):
            det.prepare(64, 64)
        assert det.provider == "cpu"
        assert det._session is cpu_ok
        assert len(det._sessions) == 1
        warnings = [r for r in caplog.records if "CoreML" in r.getMessage()]
        assert len(warnings) == 1
        # detect works through the CPU fallback session.
        frame = np.zeros((64, 64, 4), dtype=np.uint8)
        assert det.detect_bgra(frame) == []
        # A second size retry falls back again but warns only once.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="bsafe.fastdetect"):
            det.prepare(96, 96)
        assert det.provider == "cpu"
        assert det._session is cpu_ok2
        warnings = [r for r in caplog.records if "CoreML" in r.getMessage()]
        assert len(warnings) == 0


def test_prepare_raises_when_cpu_also_fails():
    import onnxruntime

    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    with (
        mock.patch.object(onnxruntime, "get_available_providers", return_value=providers),
        mock.patch.object(
            onnxruntime, "InferenceSession", side_effect=[_WarmupFailSession(), _CpuFailSession()]
        ),
    ):
        det = FullFrameNudeDetector(model_path="dummy.onnx", use_coreml=True)
        with pytest.raises(RuntimeError, match="cpu warmup boom"):
            det.prepare(64, 64)
        assert det.provider == "unprepared"
        assert det._sessions == {}


def test_postprocess_clips_left_edge():
    out, w, h = _make_output([(5, 100, 40, 20, 3, 0.9)], width=200, height=200)
    dets = postprocess(out, w, h)
    assert len(dets) == 1
    assert dets[0].box == (0, 90, 25, 20)


def test_postprocess_clips_top_edge():
    out, w, h = _make_output([(100, 5, 20, 40, 3, 0.9)], width=200, height=200)
    dets = postprocess(out, w, h)
    assert len(dets) == 1
    assert dets[0].box == (90, 0, 20, 25)


def test_postprocess_drops_fully_outside():
    out, w, h = _make_output([(-100, 100, 20, 20, 3, 0.9)], width=200, height=200)
    assert postprocess(out, w, h) == []
    out, w, h = _make_output([(100, -100, 20, 20, 3, 0.9)], width=200, height=200)
    assert postprocess(out, w, h) == []
    out, w, h = _make_output([(300, 100, 20, 20, 3, 0.9)], width=200, height=200)
    assert postprocess(out, w, h) == []


class _FakeMultiDetector:
    """Records input shapes; returns queued detections per call."""

    def __init__(self, per_call):
        self._per_call = [list(d) for d in per_call]
        self.shapes: list = []
        self.calls = 0

    def detect_bgra(self, frame):
        self.shapes.append(tuple(frame.shape))
        self.calls += 1
        idx = min(self.calls - 1, len(self._per_call) - 1)
        return list(self._per_call[idx])


def test_detect_multiscale_empty_factors_single_call():
    frame = np.zeros((100, 200, 4), dtype=np.uint8)
    primary = [Detection("FEET_EXPOSED", 0.9, (10, 10, 20, 20))]
    det = _FakeMultiDetector([primary])
    out = detect_multiscale(det, frame, ())
    assert out == primary
    assert det.calls == 1
    assert det.shapes == [(100, 200, 4)]


def test_detect_multiscale_resize_dims_and_rescale():
    frame = np.zeros((100, 200, 4), dtype=np.uint8)
    primary = [Detection("FEET_EXPOSED", 0.9, (10, 10, 20, 20))]
    small = [Detection("FEET_EXPOSED", 0.8, (10, 10, 20, 20))]
    det = _FakeMultiDetector([primary, small])
    out = detect_multiscale(det, frame, (0.5,))
    assert det.calls == 2
    assert det.shapes[0] == (100, 200, 4)
    assert det.shapes[1] == (50, 100, 4)
    assert len(out) == 2
    assert out[0].box == (10, 10, 20, 20)
    assert out[1].box == (20, 20, 40, 40)
    assert out[1].confidence == 0.8


def test_detect_multiscale_clamps_to_frame_bounds():
    frame = np.zeros((100, 200, 4), dtype=np.uint8)
    det = _FakeMultiDetector([[], [Detection("FEET_EXPOSED", 0.7, (90, 40, 20, 20))]])
    out = detect_multiscale(det, frame, (0.5,))
    assert len(out) == 1
    assert out[0].box == (180, 80, 20, 20)


def test_detect_multiscale_primary_first_ordering():
    frame = np.zeros((64, 64, 4), dtype=np.uint8)
    primary = [Detection("FEET_EXPOSED", 0.9, (1, 1, 4, 4))]
    extra = [Detection("FEET_EXPOSED", 0.5, (2, 2, 4, 4))]
    det = _FakeMultiDetector([primary, extra])
    out = detect_multiscale(det, frame, (0.5,))
    assert [d.confidence for d in out] == [0.9, 0.5]


def test_detect_multiscale_bgra_view_and_min_size():
    frame = np.zeros((40, 40, 4), dtype=np.uint8)
    view = frame[::1, ::1, :]
    assert view.shape == (40, 40, 4)
    det = _FakeMultiDetector([[], []])
    out = detect_multiscale(det, view, (0.5,))
    assert out == []
    assert det.shapes[1] == (32, 32, 4)


def test_extra_size_matches_detect_multiscale_formula():
    assert extra_size(200, 100, 0.5) == (100, 50)
    assert extra_size(200, 100, 0.5) == (
        max(32, round(200 * 0.5)),
        max(32, round(100 * 0.5)),
    )
    assert extra_size(40, 40, 0.5) == (32, 32)
    assert extra_size(1728, 1117, 0.5) == (864, 558)


def test_prepare_multiscale_warms_primary_and_extras():
    class _FakePreparer:
        def __init__(self):
            self.calls: list = []

        def prepare(self, w, h):
            self.calls.append((w, h))

    det = _FakePreparer()
    prepare_multiscale(det, 200, 100, (0.5,))
    assert det.calls[0] == (200, 100)
    assert det.calls[1:] == [extra_size(200, 100, 0.5)]
    # Empty/None factors warm only the primary shape.
    det2 = _FakePreparer()
    prepare_multiscale(det2, 200, 100, ())
    assert det2.calls == [(200, 100)]
    det3 = _FakePreparer()
    prepare_multiscale(det3, 200, 100, None)
    assert det3.calls == [(200, 100)]
    # No prepare method (e.g. EraX) is a no-op.
    prepare_multiscale(object(), 200, 100, (0.5,))


def test_prepare_multiscale_matches_detect_multiscale_sizes():
    class _FakePreparer:
        def __init__(self):
            self.calls: list = []

        def prepare(self, w, h):
            self.calls.append((w, h))

    det = _FakePreparer()
    prepare_multiscale(det, 200, 100, (0.5, 0.25))
    frame = np.zeros((100, 200, 4), dtype=np.uint8)
    multi = _FakeMultiDetector([[], [], []])
    detect_multiscale(multi, frame, (0.5, 0.25))
    # detect shapes are (h, w, c); prepare calls are (w, h).
    inferred = [(s[1], s[0]) for s in multi.shapes[1:]]
    assert inferred == det.calls[1:]


def test_detect_multiscale_extra_failure_returns_primary(caplog):
    import bsafe.fastdetect as fd_mod

    fd_mod._last_extra_warn_s = float("-inf")
    frame = np.zeros((64, 64, 4), dtype=np.uint8)
    primary = [Detection("FEET_EXPOSED", 0.9, (10, 10, 20, 20))]

    class _FailExtra:
        def __init__(self):
            self.calls = 0

        def detect_bgra(self, f):
            self.calls += 1
            if self.calls == 1:
                return list(primary)
            raise RuntimeError("extra boom")

    with caplog.at_level(logging.WARNING, logger="bsafe.fastdetect"):
        out = detect_multiscale(_FailExtra(), frame, (0.5,))
    assert out == primary
    assert any("extra" in r.getMessage().lower() for r in caplog.records)


def test_detect_multiscale_extra_resize_failure_returns_primary(caplog, monkeypatch):
    import bsafe.fastdetect as fd_mod

    fd_mod._last_extra_warn_s = float("-inf")
    frame = np.zeros((64, 64, 4), dtype=np.uint8)
    primary = [Detection("FEET_EXPOSED", 0.9, (10, 10, 20, 20))]
    det = _FakeMultiDetector([primary])
    monkeypatch.setattr(cv2, "resize", mock.Mock(side_effect=RuntimeError("resize boom")))
    with caplog.at_level(logging.WARNING, logger="bsafe.fastdetect"):
        out = detect_multiscale(det, frame, (0.5,))
    assert out == primary
    assert any("extra" in r.getMessage().lower() for r in caplog.records)


def test_detect_multiscale_extra_warning_rate_limited(caplog, monkeypatch):
    import bsafe.fastdetect as fd_mod

    fd_mod._last_extra_warn_s = float("-inf")
    times = [100.0]
    monkeypatch.setattr("bsafe.fastdetect.time.monotonic", lambda: times[0])
    frame = np.zeros((64, 64, 4), dtype=np.uint8)
    primary = [Detection("FEET_EXPOSED", 0.9, (10, 10, 20, 20))]

    class _FailExtra:
        def detect_bgra(self, f):
            raise RuntimeError("extra boom")

    class _PrimaryThenFail:
        def __init__(self):
            self._inner = _FailExtra()
            self.calls = 0

        def detect_bgra(self, f):
            self.calls += 1
            if self.calls == 1:
                return list(primary)
            return self._inner.detect_bgra(f)

    with caplog.at_level(logging.WARNING, logger="bsafe.fastdetect"):
        assert detect_multiscale(_PrimaryThenFail(), frame, (0.5,)) == primary
        assert len([r for r in caplog.records if "extra" in r.getMessage().lower()]) == 1
        caplog.clear()
        # Within 5 s the second failure is suppressed.
        times[0] = 102.0
        assert detect_multiscale(_PrimaryThenFail(), frame, (0.5,)) == primary
        assert len([r for r in caplog.records if "extra" in r.getMessage().lower()]) == 0
        caplog.clear()
        # After 5 s it warns again.
        times[0] = 106.0
        assert detect_multiscale(_PrimaryThenFail(), frame, (0.5,)) == primary
        assert len([r for r in caplog.records if "extra" in r.getMessage().lower()]) == 1
    fd_mod._last_extra_warn_s = float("-inf")


def test_detect_multiscale_primary_failure_propagates():
    import bsafe.fastdetect as fd_mod

    fd_mod._last_extra_warn_s = float("-inf")

    class _FailPrimary:
        def detect_bgra(self, frame):
            raise RuntimeError("primary boom")

    with pytest.raises(RuntimeError, match="primary boom"):
        detect_multiscale(_FailPrimary(), np.zeros((32, 32, 4), dtype=np.uint8), (0.5,))
