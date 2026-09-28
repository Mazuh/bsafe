"""Tests for the offline replay benchmark (bsafe.bench).

Synthetic only: no model, no video decoding, no Swift binary.
"""

import json
from pathlib import Path

import pytest

from bsafe import bench
from bsafe.cli import _build_parser, cmd_bench

FEET = "FEET_EXPOSED"
BREAST = "FEMALE_BREAST_EXPOSED"
FEET_ONLY = frozenset({FEET})
BOTH = frozenset({FEET, BREAST})

LEAD = 0.016


# --- Fixtures (handcrafted JSONL, never real pixels) ------------------------


def _write_dets(path, frames, *, width=200, height=100, fps=60, video="/tmp/rec.mov"):
    lines = [
        json.dumps(
            {
                "type": "header",
                "video": video,
                "width": width,
                "height": height,
                "fps": fps,
                "frames": len(frames),
                "model": "320n",
            }
        )
    ]
    for i, dets in enumerate(frames):
        payload = {
            "frame": i,
            "pts": i / fps,
            "detect_s": 0.012,
            "dets": [{"cls": cls, "conf": conf, "box": list(box)} for cls, conf, box in dets],
        }
        lines.append(json.dumps(payload))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def _write_displayed(path, frames, *, lead=LEAD, fps=60):
    lines = []
    for i, boxes in enumerate(frames):
        payload = {
            "frame": i,
            "pts": i / fps,
            "display_pts": i / fps + lead,
            "boxes": [list(b) for b in boxes],
            "applied_seq": i - 1 if i else None,
        }
        lines.append(json.dumps(payload))
    summary = {
        "type": "summary",
        "requests": len(frames),
        "applied": max(0, len(frames) - 1),
        "ignored": 0,
        "mean_latency_s": 0.0,
    }
    lines.append(json.dumps(summary))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def _fake_replay(video, boxes_path, displayed_path, **kwargs):
    """Stand-in for the Swift binary: echoes the boxes as the displayed frame."""
    _summary, records = bench.read_displayed(boxes_path)
    lines = []
    for rec in records:
        payload = {
            "frame": rec.frame,
            "pts": rec.pts,
            "display_pts": rec.pts + LEAD,
            "boxes": [list(b) for b in rec.boxes],
            "applied_seq": None,
        }
        lines.append(json.dumps(payload))
    summary = {
        "type": "summary",
        "requests": len(records),
        "applied": len(records),
        "ignored": 0,
        "mean_latency_s": 0.0,
    }
    lines.append(json.dumps(summary))
    Path(displayed_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


# --- Linking and gap fill ---------------------------------------------------


def _dets_by_frame(spec):
    return {frame: [(FEET, box)] for frame, box in spec.items()}


def test_link_objects_merges_consecutive_frames():
    spec = {0: (0, 0, 40, 40), 1: (0, 0, 40, 40), 2: (0, 0, 40, 40)}
    tracks = bench.link_objects(_dets_by_frame(spec), gap_frames=15)
    assert len(tracks) == 1
    assert tracks[0].frames == [0, 1, 2]


def test_link_objects_fills_three_frame_gap():
    spec = {f: (0, 0, 40, 40) for f in (0, 1, 2, 3, 4)}
    spec.update({f: (0, 0, 40, 40) for f in (8, 9, 10)})
    tracks = bench.link_objects(_dets_by_frame(spec), gap_frames=15)
    assert len(tracks) == 1
    filled = bench.fill_gaps(tracks[0])
    assert filled.frames == list(range(11))
    assert filled.boxes[5] == (0, 0, 40, 40)


def test_link_objects_splits_on_long_gap():
    spec = {f: (0, 0, 40, 40) for f in range(5)}
    spec.update({f: (0, 0, 40, 40) for f in range(35, 40)})
    tracks = bench.link_objects(_dets_by_frame(spec), gap_frames=15)
    assert len(tracks) == 2
    assert tracks[0].frames == [0, 1, 2, 3, 4]
    assert tracks[1].frames == [35, 36, 37, 38, 39]


def test_link_objects_respects_iou_threshold():
    near = _dets_by_frame({0: (0, 0, 40, 40), 1: (0, 0, 40, 40)})
    far = _dets_by_frame({0: (0, 0, 40, 40), 1: (200, 0, 8, 8)})
    assert len(bench.link_objects(near, gap_frames=15)) == 1
    assert len(bench.link_objects(far, gap_frames=15)) == 2


def test_link_objects_keeps_classes_apart():
    spec = {0: [(FEET, (0, 0, 40, 40)), (BREAST, (0, 0, 40, 40))]}
    spec[1] = [(FEET, (0, 0, 40, 40)), (BREAST, (0, 0, 40, 40))]
    tracks = bench.link_objects(spec, gap_frames=15)
    assert len(tracks) == 2
    # Deterministic order: first frame, then class name, then box.
    assert [t.cls for t in tracks] == [FEET, BREAST]


def test_fill_gaps_interpolates_linearly():
    track = bench.Track(FEET, {0: (0, 0, 40, 40), 4: (40, 0, 40, 40)})
    filled = bench.fill_gaps(track)
    assert filled.boxes[1] == pytest.approx((10, 0, 40, 40))
    assert filled.boxes[2] == pytest.approx((20, 0, 40, 40))
    assert filled.boxes[3] == pytest.approx((30, 0, 40, 40))
    assert track.boxes == {0: (0, 0, 40, 40), 4: (40, 0, 40, 40)}


def test_link_objects_empty_input():
    assert bench.link_objects({}, gap_frames=15) == []


# --- Coverage rasterization -------------------------------------------------


def test_coverage_fraction_full_and_half():
    box = (0, 0, 80, 80)
    assert bench.coverage_fraction(box, [(0, 0, 80, 80)]) == 1.0
    assert bench.coverage_fraction(box, [(0, 0, 40, 80)]) == pytest.approx(0.5)
    assert bench.coverage_fraction(box, [(40, 0, 40, 80)]) == pytest.approx(0.5)
    assert bench.coverage_fraction(box, []) == 0.0


def test_coverage_fraction_union_covers_whole_box():
    box = (0, 0, 80, 80)
    both = [(0, 0, 40, 80), (40, 0, 40, 80)]
    assert bench.coverage_fraction(box, both) == 1.0


def test_coverage_fraction_offset_display_box():
    box = (100, 100, 40, 40)
    assert bench.coverage_fraction(box, [(80, 80, 80, 80)]) == 1.0
    assert bench.coverage_fraction(box, [(140, 100, 40, 40)]) == 0.0


def test_excess_fraction_counts_only_far_off_area():
    shown = [(148, 60, 20, 20)]
    truth = [bench.dilate_box((10, 10, 40, 40))]
    assert bench.excess_fraction(shown, truth, 200, 100) == pytest.approx(0.02)
    assert bench.excess_fraction([(10, 10, 40, 40)], truth, 200, 100) == 0.0


def test_dilate_box_expands_centred():
    assert bench.dilate_box((10, 10, 40, 40)) == pytest.approx((-10, -10, 80, 80))


def test_over_censor_pct_averages_over_displayed_frames():
    gt = {0: [(10, 10, 40, 40)], 1: [(10, 10, 40, 40)]}
    disp_pts = [0.0, 1 / 60]
    disp_boxes = [((148, 60, 20, 20),), ()]
    pct = bench.over_censor_pct(gt, disp_pts, disp_boxes, 200, 100, lambda f: f / 60)
    # Only the first displayed frame has boxes, and 2% of the frame is far off.
    assert pct == pytest.approx(1.0)


# --- Flicker and display matching -------------------------------------------


def test_count_flickers():
    assert bench.count_flickers([]) == 0
    assert bench.count_flickers([False, False, True, True]) == 0
    assert bench.count_flickers([True, True]) == 0
    assert bench.count_flickers([True, False]) == 1
    assert bench.count_flickers([True, False, True, False]) == 2
    assert bench.count_flickers([False, True, False, False, True]) == 1


def test_nearest_display():
    pts = [0.0, 0.1, 0.2]
    assert bench.nearest_display(pts, -1.0) == 0
    assert bench.nearest_display(pts, 0.0) == 0
    assert bench.nearest_display(pts, 0.04) == 0
    assert bench.nearest_display(pts, 0.06) == 1
    assert bench.nearest_display(pts, 0.2) == 2
    assert bench.nearest_display(pts, 5.0) == 2


def test_on_screen_display():
    pts = [0.016, 0.033, 0.05]
    assert bench.on_screen_display(pts, 0.0) == 0
    assert bench.on_screen_display(pts, 0.016) == 0
    assert bench.on_screen_display(pts, 0.02) == 0
    assert bench.on_screen_display(pts, 0.033) == 1
    assert bench.on_screen_display(pts, 0.04) == 1
    assert bench.on_screen_display(pts, 5.0) == 2


def test_read_displayed_keeps_apply_events(tmp_path):
    path = tmp_path / "displayed.jsonl"
    lines = [
        json.dumps(
            {
                "frame": 0,
                "pts": 0.01,
                "display_pts": 0.026,
                "boxes": [[10, 10, 40, 40]],
                "applied_seq": 0,
                "event": "apply",
            }
        ),
        json.dumps(
            {
                "frame": 1,
                "pts": 1 / 60,
                "display_pts": 1 / 60 + LEAD,
                "boxes": [[10, 10, 40, 40]],
                "applied_seq": 0,
            }
        ),
        json.dumps({"type": "summary", "requests": 1, "applied": 1}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _summary, records = bench.read_displayed(path)
    assert len(records) == 2
    assert records[0].event == "apply"
    assert records[1].event is None


def test_over_censor_charges_frames_without_gt():
    # GT only on frame 0; displayed boxes persist on frames 1-2 with no GT.
    gt = {0: [(10, 10, 40, 40)]}
    disp_pts = [0.0, 1 / 60, 2 / 60]
    far = ((148, 60, 20, 20),)
    disp_boxes = [far, far, far]

    def pts_of(f):
        return f / 60

    pct = bench.over_censor_pct(gt, disp_pts, disp_boxes, 200, 100, pts_of, dets_frames=[0, 1, 2])
    # Frame 0: far box outside dilated GT -> 2% excess. Frames 1-2: GT empty
    # -> the whole 2% box counts as excess. Mean = 2%.
    assert pct == pytest.approx(2.0)


def test_score_on_screen_matching_uses_latest_display(tmp_path):
    # GT at frame 1 (pts 1/60). Displays: frame 0 covers, frame 1 empty but
    # with a later display_pts than GT pts... on-screen matching must use the
    # latest display <= GT pts, not the nearest future line.
    dets = _write_dets(
        tmp_path / "dets.jsonl",
        [[(FEET, 0.9, (10, 10, 40, 40))], [(FEET, 0.9, (10, 10, 40, 40))]],
    )
    _write_displayed(
        tmp_path / "displayed.jsonl",
        [[(10, 10, 40, 40)], []],
        lead=0.0,
    )
    # Rewrite display_pts so frame 0 displays at 0.0 and frame 1 at 1.0 (far
    # future): GT frame 1 at 1/60 should still see frame 0's boxes on screen.
    lines = (tmp_path / "displayed.jsonl").read_text(encoding="utf-8").splitlines()
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    rec0["display_pts"] = 0.0
    rec1["display_pts"] = 1.0
    (tmp_path / "displayed.jsonl").write_text(
        "\n".join([json.dumps(rec0), json.dumps(rec1), lines[2]]) + "\n",
        encoding="utf-8",
    )
    overall = bench.score(dets, tmp_path / "displayed.jsonl", BOTH)["overall"]
    assert overall["leak_frac"] == pytest.approx(0.0)


def test_score_per_class_over_censor_is_null(tmp_path):
    dets = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 2)
    result = bench.score(dets, disp, BOTH)
    assert isinstance(result["overall"]["over_censor_pct"], float)
    assert result["per_class"][FEET]["over_censor_pct"] is None


def test_first_cover_penalizes_never_covered(tmp_path):
    dets = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(3)])
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[] for _ in range(3)])
    overall = bench.score(dets, disp, BOTH)["overall"]
    # (last_pts - first_pts + 1 frame) = 3/60 s = 50 ms.
    assert overall["first_cover_ms"] == pytest.approx(50.0)
    assert overall["first_cover_p90_ms"] == pytest.approx(50.0)


def test_first_cover_p90(tmp_path):
    # Two objects: one covered at once, one never covered (penalty 50 ms for
    # a 3-frame clip at 60 fps). Median of [0, 50] = 25, p90 = 50.
    frames = [
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
    ]
    dets = _write_dets(tmp_path / "dets.jsonl", frames)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 3)
    result = bench.score(dets, disp, BOTH)
    assert result["overall"]["first_cover_ms"] == pytest.approx(25.0)
    assert result["overall"]["first_cover_p90_ms"] == pytest.approx(50.0)


def test_detect_pts_from_pos_msec(tmp_path, monkeypatch):
    import numpy as np

    import bsafe.bench as bench_mod

    frames_in = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

    class FakeCap:
        def __init__(self, *a, **k):
            self._i = -1
            self.msecs = [0.0, 33.0, 66.0]

        def isOpened(self):
            return True

        def get(self, prop):
            import cv2 as cv2_mod

            if prop == cv2_mod.CAP_PROP_FPS:
                return 60.0
            if prop == cv2_mod.CAP_PROP_POS_MSEC:
                return self.msecs[max(0, self._i)]
            return 0.0

        def read(self):
            self._i += 1
            if self._i < len(frames_in):
                return True, frames_in[self._i]
            return False, None

        def release(self):
            pass

    class FakeDetector:
        provider = "fake"

        def __init__(self, *a, **k):
            pass

        def prepare(self, w, h):
            pass

        def detect_bgra(self, frame):
            return []

        def close(self):
            pass

    monkeypatch.setattr(bench_mod.cv2, "VideoCapture", FakeCap)
    import bsafe.fastdetect as fastdetect_mod

    monkeypatch.setattr(fastdetect_mod, "FullFrameNudeDetector", FakeDetector)

    out = tmp_path / "out"
    # Fake file must exist for the is_file check.
    (tmp_path / "rec.mov").write_bytes(b"x")
    dets_path = bench_mod.detect(tmp_path / "rec.mov", out)
    header, records = bench_mod.read_dets(dets_path)
    assert [r.pts for r in records] == pytest.approx([0.0, 0.033, 0.066])
    assert float(header["fps"]) == pytest.approx(2 / 0.066, rel=1e-3)


def test_detect_pts_falls_back_when_msec_missing(tmp_path, monkeypatch):
    import numpy as np

    import bsafe.bench as bench_mod

    frames_in = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

    class FakeCap:
        def __init__(self, *a, **k):
            self._i = -1

        def isOpened(self):
            return True

        def get(self, prop):
            import cv2 as cv2_mod

            if prop == cv2_mod.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def read(self):
            self._i += 1
            if self._i < len(frames_in):
                return True, frames_in[self._i]
            return False, None

        def release(self):
            pass

    class FakeDetector:
        provider = "fake"

        def __init__(self, *a, **k):
            pass

        def prepare(self, w, h):
            pass

        def detect_bgra(self, frame):
            return []

        def close(self):
            pass

    monkeypatch.setattr(bench_mod.cv2, "VideoCapture", FakeCap)
    import bsafe.fastdetect as fastdetect_mod

    monkeypatch.setattr(fastdetect_mod, "FullFrameNudeDetector", FakeDetector)
    (tmp_path / "rec.mov").write_bytes(b"x")
    dets_path = bench_mod.detect(tmp_path / "rec.mov", tmp_path / "out")
    _header, records = bench_mod.read_dets(dets_path)
    assert [r.pts for r in records] == pytest.approx([0.0, 1 / 30, 2 / 30])


# --- write_boxes ------------------------------------------------------------


def test_write_boxes_pads_merges_and_filters(tmp_path):
    dets = _write_dets(
        tmp_path / "dets.jsonl",
        [
            [
                (FEET, 0.9, (10, 10, 40, 40)),
                (FEET, 0.9, (12, 12, 40, 40)),
                ("FACE_FEMALE", 0.95, (0, 0, 10, 10)),
            ],
            [],
        ],
    )
    plain = bench.write_boxes(dets, tmp_path / "boxes0.jsonl", FEET_ONLY, 0.0)
    padded = bench.write_boxes(dets, tmp_path / "boxes1.jsonl", FEET_ONLY, 0.25)

    lines = plain.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header == {"type": "header", "width": 200, "height": 100, "frames": 2}
    assert json.loads(lines[1])["boxes"] == [[10, 10, 42, 42]]
    assert json.loads(lines[2])["boxes"] == []
    assert json.loads(lines[1])["pts"] == pytest.approx(0.0)

    padded_lines = padded.read_text(encoding="utf-8").splitlines()
    assert json.loads(padded_lines[1])["boxes"] == [[0, 0, 62, 62]]


def test_write_boxes_min_padding_floor(tmp_path):
    dets = _write_dets(
        tmp_path / "dets.jsonl",
        [[(FEET, 0.9, (100, 10, 20, 20))]],
    )
    out = bench.write_boxes(dets, tmp_path / "boxes-mpx.jsonl", FEET_ONLY, 0.0, 48)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1])["boxes"] == [[52, 0, 116, 78]]
    # Default 0 reproduces the legacy output.
    legacy = bench.write_boxes(dets, tmp_path / "boxes-legacy.jsonl", FEET_ONLY, 0.0)
    assert json.loads(legacy.read_text(encoding="utf-8").splitlines()[1])["boxes"] == [
        [100, 10, 20, 20]
    ]


def test_write_boxes_requires_frame_size(tmp_path):
    dets = tmp_path / "dets.jsonl"
    dets.write_text(json.dumps({"type": "header", "frames": 0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frame size"):
        bench.write_boxes(dets, tmp_path / "boxes.jsonl", FEET_ONLY, 0.0)


# --- score() end to end -----------------------------------------------------


def test_score_perfect_display(tmp_path):
    dets = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(3)])
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 3)

    result = bench.score(dets, disp, BOTH)
    overall = result["overall"]
    assert overall["leak_frac"] == pytest.approx(0.0)
    assert overall["leak_frames_pct"] == pytest.approx(0.0)
    assert overall["flicker_per_min"] == pytest.approx(0.0)
    assert overall["first_cover_ms"] == pytest.approx(0.0)
    assert overall["over_censor_pct"] == pytest.approx(0.0)
    assert overall["n_objects"] == 1
    assert overall["n_gt_boxes"] == 3
    assert result["per_class"][FEET]["leak_frac"] == pytest.approx(0.0)


def test_score_empty_display_is_total_leak(tmp_path):
    dets = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(3)])
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[] for _ in range(3)])

    overall = bench.score(dets, disp, BOTH)["overall"]
    assert overall["leak_frac"] == pytest.approx(1.0)
    assert overall["leak_frames_pct"] == pytest.approx(100.0)
    # Never-covered objects are penalized with the full object span + 1 frame.
    assert overall["first_cover_ms"] == pytest.approx(50.0)
    assert overall["first_cover_p90_ms"] == pytest.approx(50.0)
    assert overall["over_censor_pct"] == pytest.approx(0.0)


def test_score_counts_flicker_when_display_drops_out(tmp_path):
    dets = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(3)])
    disp = _write_displayed(
        tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)], [(10, 10, 40, 40)], []], lead=0.0
    )

    overall = bench.score(dets, disp, BOTH)["overall"]
    assert overall["leak_frac"] == pytest.approx(1 / 3)
    assert overall["leak_frames_pct"] == pytest.approx(100 / 3)
    # One covered -> uncovered transition over a 3-frame clip at 60 fps.
    assert overall["flicker_per_min"] == pytest.approx(1 / (3 / 60 / 60))
    assert overall["first_cover_ms"] == pytest.approx(0.0)


def test_score_ignores_low_confidence_and_unlisted_classes(tmp_path):
    frames = [[(FEET, 0.1, (10, 10, 40, 40)), ("FACE_FEMALE", 0.99, (10, 10, 40, 40))]]
    dets = _write_dets(tmp_path / "dets.jsonl", frames)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[]])

    result = bench.score(dets, disp, BOTH)
    assert result["overall"]["n_gt_boxes"] == 0
    assert result["overall"]["n_objects"] == 0
    assert result["per_class"] == {}


def test_score_splits_per_class(tmp_path):
    frames = [
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
    ]
    dets = _write_dets(tmp_path / "dets.jsonl", frames)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 2)

    result = bench.score(dets, disp, BOTH)
    assert result["overall"]["n_gt_boxes"] == 4
    assert result["per_class"][FEET]["leak_frac"] == pytest.approx(0.0)
    assert result["per_class"][BREAST]["leak_frac"] == pytest.approx(1.0)
    assert result["overall"]["leak_frac"] == pytest.approx(0.5)


def test_score_gap_fills_missing_frames(tmp_path):
    # Endpoints must overlap with IoU >= 0.3 to link per contract (dx <= ~21px
    # for 40px boxes); (10,10,40,40) vs (30,10,40,40) gives IoU ~0.33.
    frames = [[(FEET, 0.9, (10, 10, 40, 40))], [], [(FEET, 0.9, (30, 10, 40, 40))]]
    dets = _write_dets(tmp_path / "dets.jsonl", frames)
    # The display follows the interpolated position, so every GT box is covered.
    disp = _write_displayed(
        tmp_path / "displayed.jsonl",
        [[(10, 10, 40, 40)], [(20, 10, 40, 40)], [(30, 10, 40, 40)]],
        lead=0.0,
    )

    overall = bench.score(dets, disp, BOTH)["overall"]
    assert overall["n_objects"] == 1
    assert overall["n_gt_boxes"] == 3
    assert overall["leak_frac"] == pytest.approx(0.0)


def test_score_uses_fps_from_header(tmp_path):
    dets = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (0, 0, 40, 40))]], fps=30)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(0, 0, 40, 40)]], fps=30)
    overall = bench.score(dets, disp, BOTH)["overall"]
    assert overall["n_gt_boxes"] == 1
    assert overall["leak_frac"] == pytest.approx(0.0)


# --- Replay plumbing --------------------------------------------------------


def test_replay_binary_missing_gives_build_hint(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        bench.replay_binary(tmp_path)
    message = str(excinfo.value)
    assert "swift build -c release" in message
    assert "bsafe-replay" in message


def test_run_replay_passes_contract_flags(tmp_path):
    args_file = tmp_path / "args.txt"
    script = tmp_path / "fake-replay"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{args_file}"\n'
        'echo "Warning: video has 1 extra trailing frame" >&2\n'
        "exit 0\n"
    )
    script.chmod(0o755)

    out = tmp_path / "displayed.jsonl"
    code = bench.run_replay(tmp_path / "rec.mov", tmp_path / "boxes.jsonl", out, binary=script)
    assert code == 0
    # "$@" excludes argv[0] (the binary path), so the file holds only flags.
    flags = args_file.read_text(encoding="utf-8").splitlines()
    assert flags[:6] == [
        "--video",
        str(tmp_path / "rec.mov"),
        "--boxes",
        str(tmp_path / "boxes.jsonl"),
        "--out",
        str(out),
    ]
    assert flags[6:] == [
        "--present-lead-ms",
        "16",
        "--overhead-ms",
        "6",
        "--persist-passes",
        "8",
        "--shrink-alpha",
        "0.1",
        "--min-persist-s",
        "1",
        "--max-lead-s",
        "0.1",
    ]


def test_run_replay_raises_on_failure(tmp_path, capsys):
    script = tmp_path / "fake-replay"
    script.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match="boom"):
        bench.run_replay(
            tmp_path / "rec.mov", tmp_path / "b.jsonl", tmp_path / "d.jsonl", binary=script
        )
    assert "boom" in capsys.readouterr().err


def test_detect_rejects_non_nudenet_model(tmp_path):
    with pytest.raises(ValueError, match="NudeNet models only"):
        bench.detect(tmp_path / "rec.mov", tmp_path / "out", model="erax-nano")


def test_detect_requires_existing_video(tmp_path):
    with pytest.raises(FileNotFoundError, match="video not found"):
        bench.detect(tmp_path / "nope.mov", tmp_path / "out", model="320n")


# --- Sweep ------------------------------------------------------------------


def test_config_name_is_stable():
    assert bench.config_name(0.0, 8, 0.6, 0.25) == "pad0-pp8-mp0.6-sa0.25"
    assert bench.config_name(0.4, 8, 1.0, 0.1) == "pad0.4-pp8-mp1-sa0.1"


def test_config_name_min_padding_suffix():
    assert bench.config_name(0.4, 8, 0.6, 0.25) == "pad0.4-pp8-mp0.6-sa0.25"
    assert bench.config_name(0.4, 8, 0.6, 0.25, 48) == "pad0.4-pp8-mp0.6-sa0.25-mpx48"
    assert bench.config_name(0.4, 8, 0.6, 0.25, 0) == "pad0.4-pp8-mp0.6-sa0.25"


def test_sweep_runs_grid_and_writes_sweep_json(tmp_path, monkeypatch, capsys):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(3)])
    monkeypatch.setattr(bench, "run_replay", _fake_replay)

    payload = bench.sweep(tmp_path)

    assert len(payload["results"]) == 12
    assert payload["censor"] == "body"
    leaks = [entry["overall"]["leak_frac"] for entry in payload["results"]]
    assert leaks == sorted(leaks)
    assert all(leak == pytest.approx(0.0) for leak in leaks)

    on_disk = json.loads((tmp_path / "sweep.json").read_text(encoding="utf-8"))
    assert on_disk["video"] == "/tmp/rec.mov"
    assert len(on_disk["results"]) == 12
    names = [entry["name"] for entry in on_disk["results"]]
    assert "pad0-pp8-mp0.6-sa0.25" in names
    assert "pad0.4-pp8-mp1-sa0.1" in names

    for entry in on_disk["results"]:
        run_dir = tmp_path / entry["name"]
        assert (run_dir / "boxes.jsonl").is_file()
        assert (run_dir / "displayed.jsonl").is_file()
        score = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))
        assert set(score) == {"config", "overall", "per_class"}
        assert score["config"]["censor"] == "body"

    out = capsys.readouterr().out
    assert "leak_frac" in out
    assert "over_censor%" in out
    assert "pad0.4-pp8-mp0.6-sa0.25" in out


def test_sweep_requires_dets(tmp_path):
    with pytest.raises(FileNotFoundError, match="dets.jsonl"):
        bench.sweep(tmp_path)


def test_sweep_rejects_unknown_censor(tmp_path):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]])
    with pytest.raises(ValueError, match="unknown censor preset"):
        bench.sweep(tmp_path, "bogus")


# --- CLI --------------------------------------------------------------------


def _args(extra=None):
    parser, *_ = _build_parser()
    return parser.parse_args(["bench", *(extra or [])])


def test_parser_bench_detect_defaults():
    args = _args(["detect", "rec.mov", "-o", "out"])
    assert args.command == "bench"
    assert args.bench_command == "detect"
    assert args.input == "rec.mov"
    assert args.output == "out"
    assert args.model == "320n"
    assert args.extra_scales is None


def test_parser_bench_replay_defaults():
    args = _args(["replay", "dir"])
    assert args.bench_command == "replay"
    assert args.dir == "dir"
    assert args.censor == "body"
    assert args.padding == 0.4
    assert args.min_padding == 0
    assert args.persist_passes == 8
    assert args.min_persist_s == 1.0
    assert args.shrink_alpha == 0.1
    assert args.present_lead_ms == 16.0
    assert args.overhead_ms == 6.0
    assert args.name is None


def test_parser_bench_replay_overrides():
    args = _args(
        [
            "replay",
            "dir",
            "--censor",
            "all",
            "--padding",
            "0.2",
            "--persist-passes",
            "4",
            "--min-persist-s",
            "1.0",
            "--shrink-alpha",
            "0.1",
            "--present-lead-ms",
            "20",
            "--overhead-ms",
            "8",
            "--name",
            "run1",
        ]
    )
    assert args.censor == "all"
    assert args.padding == 0.2
    assert args.persist_passes == 4
    assert args.min_persist_s == 1.0
    assert args.shrink_alpha == 0.1
    assert args.present_lead_ms == 20.0
    assert args.overhead_ms == 8.0
    assert args.name == "run1"


def test_parser_bench_sweep_defaults():
    args = _args(["sweep", "dir"])
    assert args.bench_command == "sweep"
    assert args.dir == "dir"
    assert args.censor == "body"
    assert args.min_padding == 0


def test_parser_bench_rejects_bad_input():
    parser, *_ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bench"])
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "detect"])
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "detect", "rec.mov"])
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "replay", "dir", "--censor", "bogus"])
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "detect", "rec.mov", "-o", "out", "--model", "erax-nano"])


def test_cmd_bench_dispatches(monkeypatch):
    seen = []
    monkeypatch.setattr(bench, "cmd_replay", lambda a: seen.append(a.bench_command))
    cmd_bench(_args(["replay", "dir"]))
    assert seen == ["replay"]


def test_cmd_replay_writes_score(tmp_path, monkeypatch, capsys):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(3)])
    monkeypatch.setattr(bench, "run_replay", _fake_replay)

    bench.cmd_replay(_args(["replay", str(tmp_path), "--padding", "0.25"]))

    run_dir = tmp_path / "pad0.25-pp8-mp1-sa0.1"
    assert (run_dir / "boxes.jsonl").is_file()
    assert (run_dir / "displayed.jsonl").is_file()
    payload = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))
    assert payload["config"]["padding"] == 0.25
    assert payload["overall"]["leak_frac"] == pytest.approx(0.0)
    out = capsys.readouterr().out
    assert "leak_frac" in out


def test_cmd_replay_reports_missing_dets(tmp_path, capsys):
    with pytest.raises(SystemExit):
        bench.cmd_replay(_args(["replay", str(tmp_path)]))
    assert "Error:" in capsys.readouterr().err


def test_cmd_sweep_runs_grid(tmp_path, monkeypatch):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    monkeypatch.setattr(bench, "run_replay", _fake_replay)
    bench.cmd_sweep(_args(["sweep", str(tmp_path)]))
    assert (tmp_path / "sweep.json").is_file()


def test_cmd_replay_records_min_padding(tmp_path, monkeypatch):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    monkeypatch.setattr(bench, "run_replay", _fake_replay)
    bench.cmd_replay(_args(["replay", str(tmp_path), "--min-padding", "48"]))
    run_dir = tmp_path / "pad0.4-pp8-mp1-sa0.1-mpx48"
    assert (run_dir / "boxes.jsonl").is_file()
    payload = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))
    assert payload["config"]["min_padding"] == 48


def test_cmd_replay_rejects_negative_min_padding(tmp_path):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]])
    with pytest.raises(SystemExit):
        bench.cmd_replay(_args(["replay", str(tmp_path), "--min-padding", "-1"]))


def test_sweep_applies_min_padding_to_all_configs(tmp_path, monkeypatch):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    monkeypatch.setattr(bench, "run_replay", _fake_replay)
    payload = bench.sweep(tmp_path, min_padding=48)
    assert len(payload["results"]) == 12
    assert all(entry["name"].endswith("-mpx48") for entry in payload["results"])
    assert all(entry["config"]["min_padding"] == 48 for entry in payload["results"])


def test_cmd_sweep_passes_min_padding(tmp_path, monkeypatch):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    monkeypatch.setattr(bench, "run_replay", _fake_replay)
    bench.cmd_sweep(_args(["sweep", str(tmp_path), "--min-padding", "48"]))
    on_disk = json.loads((tmp_path / "sweep.json").read_text(encoding="utf-8"))
    assert all(entry["name"].endswith("-mpx48") for entry in on_disk["results"])


# --- External GT (recall) -----------------------------------------------------


def test_gt_scaling_from_2x_resolution(tmp_path):
    live = _write_dets(tmp_path / "dets.jsonl", [[], []], width=200, height=100)
    gt = _write_dets(
        tmp_path / "gt.jsonl",
        [[(FEET, 0.9, (20, 20, 80, 80))], [(FEET, 0.9, (20, 20, 80, 80))]],
        width=400,
        height=200,
    )
    header, records = bench.read_dets(live)
    dets_by_frame, sources = bench._gt_dets_by_frame(
        header, records, [gt], BOTH, bench.DEFAULT_GT_CONF
    )
    assert sources == [str(gt.resolve())]
    assert dets_by_frame[0] == [(FEET, (10, 10, 40, 40))]
    assert dets_by_frame[1] == [(FEET, (10, 10, 40, 40))]
    # Scaled GT is covered by the live-scale display box.
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 2)
    overall = bench.score(live, disp, BOTH, gt_dets=[gt])["overall"]
    assert overall["leak_frac"] == pytest.approx(0.0)
    assert overall["n_objects"] == 1


def test_gt_union_duplicate_merge(tmp_path):
    live = _write_dets(tmp_path / "dets.jsonl", [[], [], []])
    box = (FEET, 0.9, (10, 10, 40, 40))
    gt1 = _write_dets(tmp_path / "gt1.jsonl", [[box], [box], [box]])
    gt2 = _write_dets(tmp_path / "gt2.jsonl", [[box], [box], [box]])
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[] for _ in range(3)])
    result = bench.score(live, disp, BOTH, gt_dets=[gt1, gt2])
    assert result["overall"]["n_objects"] == 1
    assert result["overall"]["n_gt_boxes"] == 3


def test_gt_frame_count_tolerance_warns(tmp_path, capsys):
    live = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]] * 5)
    gt = _write_dets(tmp_path / "gt.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]] * 6)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 5)
    result = bench.score(live, disp, BOTH, gt_dets=[gt])
    assert result["overall"]["n_objects"] == 1
    assert "Warning" in capsys.readouterr().err


def test_gt_frame_count_mismatch_errors(tmp_path):
    live = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]] * 5)
    gt = _write_dets(tmp_path / "gt.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]] * 8)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 5)
    with pytest.raises(ValueError, match="mismatch"):
        bench.score(live, disp, BOTH, gt_dets=[gt])


def test_objects_missed_pct(tmp_path):
    frames = [
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
    ]
    dets = _write_dets(tmp_path / "dets.jsonl", frames)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 2)
    result = bench.score(dets, disp, BOTH)
    assert result["overall"]["objects_missed_pct"] == pytest.approx(50.0)
    assert result["per_class"][FEET]["objects_missed_pct"] == pytest.approx(0.0)
    assert result["per_class"][BREAST]["objects_missed_pct"] == pytest.approx(100.0)


def test_objects_missed_by_live_pct(tmp_path):
    live = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]] * 3)
    gt_frames = [
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
        [(FEET, 0.9, (10, 10, 40, 40)), (BREAST, 0.9, (100, 10, 40, 40))],
    ]
    gt = _write_dets(tmp_path / "gt.jsonl", gt_frames)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[] for _ in range(3)])
    result = bench.score(live, disp, BOTH, gt_dets=[gt])
    assert result["overall"]["n_objects"] == 2
    assert result["overall"]["objects_missed_pct"] == pytest.approx(100.0)
    assert result["overall"]["objects_missed_by_live_pct"] == pytest.approx(50.0)
    assert result["per_class"][FEET]["objects_missed_by_live_pct"] == pytest.approx(0.0)
    assert result["per_class"][BREAST]["objects_missed_by_live_pct"] == pytest.approx(100.0)


def test_objects_missed_by_live_far_box_counts_as_missed(tmp_path):
    live = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (150, 60, 20, 20))]] * 2)
    gt = _write_dets(tmp_path / "gt.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]] * 2)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[] for _ in range(2)])
    result = bench.score(live, disp, BOTH, gt_dets=[gt])
    assert result["overall"]["objects_missed_by_live_pct"] == pytest.approx(100.0)


def test_score_without_gt_dets_has_new_metrics(tmp_path):
    dets = _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[(10, 10, 40, 40)]] * 2)
    overall = bench.score(dets, disp, BOTH)["overall"]
    assert overall["objects_missed_pct"] == pytest.approx(0.0)
    assert overall["objects_missed_by_live_pct"] == pytest.approx(0.0)


def test_gt_conf_applies_to_all_sources(tmp_path):
    live = _write_dets(tmp_path / "dets.jsonl", [[], []])
    gt1 = _write_dets(tmp_path / "gt1.jsonl", [[(FEET, 0.3, (10, 10, 40, 40))]] * 2)
    gt2 = _write_dets(tmp_path / "gt2.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))]] * 2)
    disp = _write_displayed(tmp_path / "displayed.jsonl", [[] for _ in range(2)])
    low = bench.score(live, disp, BOTH, gt_conf=0.25, gt_dets=[gt1, gt2])
    assert low["overall"]["n_objects"] == 1
    high = bench.score(live, disp, BOTH, gt_conf=0.95, gt_dets=[gt1, gt2])
    assert high["overall"]["n_objects"] == 0


def test_parser_bench_replay_gt_dets():
    args = _args(["replay", "dir", "--gt-dets", "a.jsonl", "--gt-dets", "b.jsonl"])
    assert args.gt_dets == ["a.jsonl", "b.jsonl"]
    assert args.gt_conf == pytest.approx(0.25)
    args2 = _args(["replay", "dir", "--gt-conf", "0.5"])
    assert args2.gt_conf == pytest.approx(0.5)
    assert args2.gt_dets is None


def test_parser_bench_sweep_gt_dets():
    args = _args(["sweep", "dir", "--gt-dets", "a.jsonl", "--gt-conf", "0.4"])
    assert args.gt_dets == ["a.jsonl"]
    assert args.gt_conf == pytest.approx(0.4)


def test_cmd_replay_records_gt_sources(tmp_path, monkeypatch, capsys):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    gt = _write_dets(tmp_path / "gt.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    monkeypatch.setattr(bench, "run_replay", _fake_replay)
    bench.cmd_replay(_args(["replay", str(tmp_path), "--gt-dets", str(gt)]))
    run_dir = tmp_path / "pad0.4-pp8-mp1-sa0.1"
    payload = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))
    assert payload["config"]["gt_sources"] == [str(gt.resolve())]
    assert payload["config"]["gt_conf"] == pytest.approx(0.25)
    out = capsys.readouterr().out
    assert "objects_missed" in out
    assert "objects_missed_by_live" in out


def test_sweep_forwards_gt_dets(tmp_path, monkeypatch):
    _write_dets(tmp_path / "dets.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    gt = _write_dets(tmp_path / "gt.jsonl", [[(FEET, 0.9, (10, 10, 40, 40))] for _ in range(2)])
    monkeypatch.setattr(bench, "run_replay", _fake_replay)
    payload = bench.sweep(tmp_path, gt_dets=[str(gt)], gt_conf=0.3)
    assert payload["gt_sources"] == [str(gt.resolve())]
    assert payload["gt_conf"] == pytest.approx(0.3)
    for entry in payload["results"]:
        assert entry["config"]["gt_sources"] == [str(gt.resolve())]
        assert entry["overall"]["objects_missed_pct"] == pytest.approx(0.0)


def test_parse_bench_extra_scales():
    assert bench.parse_bench_extra_scales(None) == ()
    assert bench.parse_bench_extra_scales("none") == ()
    assert bench.parse_bench_extra_scales("") == ()
    assert bench.parse_bench_extra_scales("0.5") == (0.5,)
    assert bench.parse_bench_extra_scales("0.5,0.333") == pytest.approx((0.5, 0.333))
    assert bench.parse_bench_extra_scales([0.5]) == (0.5,)
    with pytest.raises(ValueError):
        bench.parse_bench_extra_scales("1.0")
    with pytest.raises(ValueError):
        bench.parse_bench_extra_scales("0")
    with pytest.raises(ValueError):
        bench.parse_bench_extra_scales("0.5,0.5")
    with pytest.raises(ValueError):
        bench.parse_bench_extra_scales("0.1,0.2,0.3,0.4")


def _bench_fake_env(monkeypatch, frames_in, detector_cls):
    import bsafe.bench as bench_mod
    import bsafe.fastdetect as fastdetect_mod

    class FakeCap:
        def __init__(self, *a, **k):
            self._i = -1

        def isOpened(self):
            return True

        def get(self, prop):
            import cv2 as cv2_mod

            if prop == cv2_mod.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def read(self):
            self._i += 1
            if self._i < len(frames_in):
                return True, frames_in[self._i]
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(bench_mod.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(fastdetect_mod, "FullFrameNudeDetector", detector_cls)


def test_detect_records_extra_scales_in_header(tmp_path, monkeypatch):
    import numpy as np

    import bsafe.bench as bench_mod

    frames_in = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]

    class FakeDetector:
        provider = "fake"

        def __init__(self, *a, **k):
            pass

        def prepare(self, w, h):
            pass

        def detect_bgra(self, frame):
            return []

        def close(self):
            pass

    _bench_fake_env(monkeypatch, frames_in, FakeDetector)
    (tmp_path / "rec.mov").write_bytes(b"x")
    dets_path = bench_mod.detect(tmp_path / "rec.mov", tmp_path / "out", extra_scales="0.5")
    header, records = bench_mod.read_dets(dets_path)
    assert header["extra_scales"] == [0.5]
    assert len(records) == 2
    default_path = bench_mod.detect(tmp_path / "rec.mov", tmp_path / "out2")
    default_header, _ = bench_mod.read_dets(default_path)
    assert default_header["extra_scales"] == []


def test_detect_extra_scales_runs_multiscale_passes(tmp_path, monkeypatch):
    import numpy as np

    import bsafe.bench as bench_mod

    frames_in = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    shapes: list = []

    class FakeDetector:
        provider = "fake"

        def __init__(self, *a, **k):
            pass

        def prepare(self, w, h):
            pass

        def detect_bgra(self, frame):
            shapes.append(tuple(frame.shape))
            return []

        def close(self):
            pass

    _bench_fake_env(monkeypatch, frames_in, FakeDetector)
    (tmp_path / "rec.mov").write_bytes(b"x")
    bench_mod.detect(tmp_path / "rec.mov", tmp_path / "out", extra_scales=(0.5,))
    # 2 frames x (1 primary + 1 extra) passes.
    assert len(shapes) == 4
    assert shapes[0] == (64, 64, 3)
    assert shapes[1] == (32, 32, 3)


def test_cmd_detect_rejects_bad_extra_scales(tmp_path):
    with pytest.raises(SystemExit):
        bench.cmd_detect(
            _args(
                [
                    "detect",
                    str(tmp_path / "rec.mov"),
                    "-o",
                    str(tmp_path / "o"),
                    "--extra-scales",
                    "1.5",
                ]
            )
        )


def test_detect_warms_extra_shapes_before_loop(tmp_path, monkeypatch):
    import numpy as np

    import bsafe.bench as bench_mod
    from bsafe.fastdetect import extra_size

    frames_in = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    prepare_calls: list = []

    class FakeDetector:
        provider = "fake"

        def __init__(self, *a, **k):
            pass

        def prepare(self, w, h):
            prepare_calls.append((w, h))

        def detect_bgra(self, frame):
            return []

        def close(self):
            pass

    _bench_fake_env(monkeypatch, frames_in, FakeDetector)
    (tmp_path / "rec.mov").write_bytes(b"x")
    bench_mod.detect(tmp_path / "rec.mov", tmp_path / "out", extra_scales="0.5")
    assert (64, 64) in prepare_calls
    assert extra_size(64, 64, 0.5) in prepare_calls
