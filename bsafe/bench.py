"""Offline replay benchmark for the live censor: detect → replay → score.

``bsafe bench detect`` runs full-frame NudeNet on every frame of a screen
recording and writes ``DIR/dets.jsonl``. ``bsafe bench replay`` turns those
detections into censor boxes (class filter + padding + merge, same as live),
hands them to the Swift ``bsafe-replay`` binary, which replays them through the
live tracker on the live frame-credit schedule, and then scores what was on
screen against oracle ground truth. ``bsafe bench sweep`` runs a small grid of
padding/tracker configs and prints a table sorted by leak.

Only detection metadata and box coordinates are written — never pixels, crops
or frames.
"""

import json
import math
import os
import subprocess
import sys
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median

import cv2
import numpy as np

from bsafe.censor import CENSOR_PRESETS, build_censor_boxes
from bsafe.detector import Detection, get_model_backend, resolve_model
from bsafe.style import bold, dim, error, info, success, warn

# --- Contract constants -----------------------------------------------------

DETS_NAME = "dets.jsonl"
BOXES_NAME = "boxes.jsonl"
DISPLAYED_NAME = "displayed.jsonl"
SCORE_NAME = "score.json"
SWEEP_NAME = "sweep.json"

REPLAY_BINARY_NAME = "bsafe-replay"
BUILD_HINT = "cd swift && swift build -c release"

DEFAULT_FPS = 60.0
PROGRESS_EVERY = 100

# Ground truth
LINK_IOU = 0.3
DEFAULT_GT_CONF = 0.25
DEFAULT_GAP_S = 0.25

# Scoring
COVER_THRESHOLD = 0.8
GRID = 4  # coverage rasterized on a grid downscaled by this factor
GT_DILATE = 0.5  # per side, as a fraction of each GT box size
GT_MERGE_IOU = 0.5  # same-frame, same-class GT boxes merged into their union
LIVE_MATCH_IOU = 0.1  # live-vs-GT IoU to count an object as "seen" by live

# Replay defaults (TrackerConfig defaults in Swift)
DEFAULT_PRESENT_LEAD_MS = 16.0
DEFAULT_OVERHEAD_MS = 6.0
DEFAULT_PERSIST_PASSES = 8
DEFAULT_MIN_PERSIST_S = 1.0
DEFAULT_SHRINK_ALPHA = 0.1
DEFAULT_MAX_LEAD_S = 0.1
DEFAULT_CENSOR = "body"

# Sweep grid (3 x 2 x 2 = 12 configs)
SWEEP_PADDINGS = (0.0, 0.2, 0.4)
SWEEP_PERSIST = ((8, 0.6), (8, 1.0))
SWEEP_SHRINK = (0.25, 0.1)

Box = tuple[float, float, float, float]  # x, y, w, h


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One decoded frame and its detections (from dets.jsonl)."""

    frame: int
    pts: float
    detect_s: float
    dets: tuple[Detection, ...]


@dataclass(frozen=True, slots=True)
class DisplayRecord:
    """One replayed frame and the boxes that were on screen (from displayed.jsonl)."""

    frame: int
    pts: float
    display_pts: float
    boxes: tuple[Box, ...]
    applied_seq: int | None
    event: str | None = None


@dataclass(slots=True)
class Track:
    """A linked ground-truth object: one box per frame, keys are frame indices."""

    cls: str
    boxes: dict[int, Box] = field(default_factory=dict)

    @property
    def frames(self) -> list[int]:
        return sorted(self.boxes)


# --- Small helpers ----------------------------------------------------------


def _parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def _num(value: float) -> str:
    """Format a number for a command-line flag (no trailing zeros)."""
    return f"{float(value):g}"


def _json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _iou(a: Box, b: Box) -> float:
    """IoU of two (x, y, w, h) boxes."""
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 0.0
    return inter / union


def censor_classes(preset: str) -> frozenset[str]:
    """Resolve a censor preset name to its class set. Raises ValueError if unknown."""
    if preset not in CENSOR_PRESETS:
        known = ", ".join(sorted(CENSOR_PRESETS))
        raise ValueError(f"unknown censor preset '{preset}'. Known presets: {known}")
    return CENSOR_PRESETS[preset]


def require_dets(base: Path) -> Path:
    """Return the dets.jsonl path of a detect directory. Raises FileNotFoundError if absent."""
    dets_path = base / DETS_NAME
    if not dets_path.is_file():
        hint = "run 'bsafe bench detect' first"
        raise FileNotFoundError(f"{DETS_NAME} not found in {base} — {hint}")
    return dets_path


def require_video(header: dict, dets_path: Path) -> str:
    """Return the recording path stored in a dets.jsonl header. Raises ValueError if absent."""
    video = header.get("video")
    if not video:
        raise ValueError(f"{dets_path} has no 'video' path in its header — re-run bench detect")
    return str(video)


# --- JSONL I/O --------------------------------------------------------------


def read_dets(path: str | Path) -> tuple[dict, list[FrameRecord]]:
    """Read dets.jsonl. Returns (header, per-frame records in file order)."""
    header: dict = {}
    records: list[FrameRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = _parse_line(line)
            if obj is None:
                continue
            if obj.get("type") == "header":
                header = obj
                continue
            if "frame" not in obj:
                continue
            records.append(
                FrameRecord(
                    frame=int(obj["frame"]),
                    pts=float(obj.get("pts", 0.0)),
                    detect_s=float(obj.get("detect_s", 0.0)),
                    dets=tuple(_detection_from_json(d) for d in obj.get("dets", [])),
                )
            )
    return header, records


def read_displayed(path: str | Path) -> tuple[dict, list[DisplayRecord]]:
    """Read displayed.jsonl. Returns (summary, per-frame display records)."""
    summary: dict = {}
    records: list[DisplayRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = _parse_line(line)
            if obj is None:
                continue
            kind = obj.get("type")
            if kind == "summary":
                summary = obj
                continue
            if kind == "header" or "frame" not in obj:
                continue
            records.append(
                DisplayRecord(
                    frame=int(obj["frame"]),
                    pts=float(obj.get("pts", 0.0)),
                    display_pts=float(obj.get("display_pts", obj.get("pts", 0.0))),
                    boxes=tuple(tuple(float(v) for v in b) for b in obj.get("boxes", [])),
                    applied_seq=obj.get("applied_seq"),
                    event=obj.get("event"),
                )
            )
    return summary, records


def _detection_from_json(obj: dict) -> Detection:
    box = obj["box"]
    return Detection(
        class_name=str(obj["cls"]),
        confidence=float(obj["conf"]),
        box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
    )


def _dets_from_json(dets: list[Detection]) -> list[dict]:
    return [{"cls": d.class_name, "conf": round(d.confidence, 4), "box": list(d.box)} for d in dets]


# --- Step 1: detect ---------------------------------------------------------


def parse_bench_extra_scales(raw) -> tuple[float, ...]:
    """Parse bench detect --extra-scales (frame-relative factors).

    Accepts None (default, no extra passes), "none"/"" (disabled), a comma
    list string like "0.5" or "0.5,0.75", or a list/tuple of floats.
    Raises ValueError unless each factor satisfies 0 < f < 1 (at most 3,
    no duplicates).
    """
    if raw is None:
        return ()
    if isinstance(raw, (int, float)):
        factors = (float(raw),)
    elif isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return ()
        try:
            factors = tuple(float(v) for v in raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid --extra-scales {raw!r}: expected floats") from e
    else:
        s = str(raw).strip()
        if s == "" or s.lower() == "none":
            return ()
        parts = [p.strip() for p in s.split(",")]
        if any(p == "" for p in parts):
            raise ValueError(f"invalid --extra-scales {raw!r}: expected comma-separated floats")
        try:
            factors = tuple(float(p) for p in parts)
        except ValueError as e:
            raise ValueError(
                f"invalid --extra-scales {raw!r}: expected comma-separated floats"
            ) from e
    if len(factors) > 3:
        raise ValueError(f"invalid --extra-scales {list(factors)}: at most 3 values")
    if len(set(factors)) != len(factors):
        raise ValueError(f"invalid --extra-scales {list(factors)}: duplicate values")
    for f in factors:
        if not (0 < f < 1):
            raise ValueError(f"invalid --extra-scales {f}: each factor must satisfy 0 < f < 1")
    return factors


def detect(
    video: str | Path,
    out_dir: str | Path,
    model: str = "320n",
    confidence: float = 0.0,
    extra_scales=None,
) -> Path:
    """Run full-frame NudeNet on every frame of ``video``; write ``out_dir/dets.jsonl``.

    Only NudeNet models are supported. ``extra_scales`` are frame-relative
    factors (the recording is already at the live capture size); when
    non-empty each frame gets extra downscaled passes via
    ``fastdetect.detect_multiscale``. Returns the dets.jsonl path.
    """
    backend = get_model_backend(model)
    if backend != "nudenet":
        raise ValueError(f"bench supports NudeNet models only, got '{model}' ({backend})")
    factors = parse_bench_extra_scales(extra_scales)
    model_info = resolve_model(model)
    video_path = Path(video).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dets_path = out / DETS_NAME

    print(f"Loading model {bold(model)}...")
    from bsafe.fastdetect import FullFrameNudeDetector

    detector = FullFrameNudeDetector(model_path=model_info.path, min_confidence=confidence)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = DEFAULT_FPS
    ok, frame = cap.read()
    if not ok:
        cap.release()
        detector.close()
        raise RuntimeError(f"no frames decoded from {video_path}")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    print(f"Input: {bold(str(video_path))}")
    print(dim(f"  {width}x{height} px, {fps:.2f} FPS, model {model}"))

    prepare_start = time.perf_counter()
    from bsafe.fastdetect import prepare_multiscale

    prepare_multiscale(detector, width, height, factors)
    prepare_s = time.perf_counter() - prepare_start
    print(dim(f"  detector ready ({detector.provider}, {prepare_s:.2f} s)"))

    # Frames stream to a scratch file so the header can carry the exact count.
    scratch = out / (DETS_NAME + ".part")
    frames = 0
    detect_total = 0.0
    last_pts = 0.0
    # POS_MSEC for the current frame (captured right after its cap.read()).
    # First frame pts is defined as 0.
    current_msec: float | None = None
    start = time.perf_counter()
    try:
        with open(scratch, "w", encoding="utf-8") as body:
            while True:
                k = frames
                if k == 0:
                    pts = 0.0
                else:
                    pts = k / fps
                    if (
                        current_msec is not None
                        and math.isfinite(current_msec)
                        and current_msec > 0
                    ):
                        msec_s = current_msec / 1000.0
                        if math.isfinite(msec_s) and msec_s > last_pts:
                            pts = msec_s
                t0 = time.perf_counter()
                if factors:
                    from bsafe.fastdetect import detect_multiscale

                    detections = detect_multiscale(detector, frame, factors)
                else:
                    detections = detector.detect_bgra(frame)
                detect_s = time.perf_counter() - t0
                detect_total += detect_s
                frames += 1
                last_pts = pts
                payload = {
                    "frame": frames - 1,
                    "pts": pts,
                    "detect_s": round(detect_s, 6),
                    "dets": _dets_from_json(detections),
                }
                body.write(_json(payload) + "\n")
                if frames % PROGRESS_EVERY == 0:
                    _print_progress(frames, detect_total, start)
                ok, frame = cap.read()
                if not ok:
                    break
                try:
                    current_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                except Exception:
                    current_msec = None
        cap.release()
        detector.close()

        header_fps = fps
        if frames > 1 and math.isfinite(last_pts) and last_pts > 0:
            header_fps = (frames - 1) / last_pts
        header = {
            "type": "header",
            "video": str(video_path.resolve()),
            "width": width,
            "height": height,
            "fps": round(header_fps, 6),
            "frames": frames,
            "model": model,
            "extra_scales": list(factors),
        }
        with open(dets_path, "w", encoding="utf-8") as f:
            f.write(_json(header) + "\n")
            with open(scratch, encoding="utf-8") as body:
                f.write(body.read())
    finally:
        if scratch.exists():
            os.unlink(scratch)

    elapsed = time.perf_counter() - start
    avg_ms = detect_total / frames * 1000.0 if frames else 0.0
    print(
        f"{success('Saved')} {bold(str(dets_path))} "
        f"{dim(f'({frames} frames, {elapsed:.1f} s, {avg_ms:.1f} ms/frame)')}"
    )
    return dets_path


def _print_progress(frames: int, detect_total: float, start: float) -> None:
    elapsed = time.perf_counter() - start
    avg_ms = detect_total / frames * 1000.0 if frames else 0.0
    rate = frames / elapsed if elapsed > 0 else 0.0
    print(f"  {info(str(frames))} frames {dim(f'({rate:.1f}/s, {avg_ms:.1f} ms/frame)')}")


# --- Step 2a: censor boxes --------------------------------------------------


def write_boxes(
    dets_path: str | Path,
    out_path: str | Path,
    classes: frozenset[str],
    padding: float,
    min_padding: int = 0,
) -> Path:
    """Write boxes.jsonl: class-filtered, padded and merged boxes (same as live)."""
    classes = frozenset(classes)
    header, records = read_dets(dets_path)
    width = int(header.get("width") or 0)
    height = int(header.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"dets header has no frame size: {dets_path}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        header_line = {"type": "header", "width": width, "height": height, "frames": len(records)}
        f.write(_json(header_line) + "\n")
        for rec in records:
            boxes = build_censor_boxes(list(rec.dets), classes, padding, width, height, min_padding)
            payload = {
                "frame": rec.frame,
                "pts": rec.pts,
                "detect_s": rec.detect_s,
                "boxes": [list(b) for b in boxes],
            }
            f.write(_json(payload) + "\n")
    return out


# --- Step 2b: Swift replay --------------------------------------------------


def replay_binary(root: str | Path | None = None) -> Path:
    """Locate the compiled bsafe-replay binary. Raises FileNotFoundError with a build hint."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    path = base / "swift" / ".build" / "release" / REPLAY_BINARY_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{REPLAY_BINARY_NAME} not found at {path}\nBuild it with:\n  {BUILD_HINT}"
        )
    return path


def run_replay(
    video: str | Path,
    boxes_path: str | Path,
    displayed_path: str | Path,
    *,
    present_lead_ms: float = DEFAULT_PRESENT_LEAD_MS,
    overhead_ms: float = DEFAULT_OVERHEAD_MS,
    persist_passes: int = DEFAULT_PERSIST_PASSES,
    min_persist_s: float = DEFAULT_MIN_PERSIST_S,
    shrink_alpha: float = DEFAULT_SHRINK_ALPHA,
    max_lead_s: float = DEFAULT_MAX_LEAD_S,
    binary: str | Path | None = None,
) -> int:
    """Run the Swift replay binary over boxes.jsonl. Writes displayed.jsonl.

    Raises RuntimeError (with the tail of the helper's stderr) on a non-zero exit.
    """
    exe = Path(binary) if binary is not None else replay_binary()
    out = Path(displayed_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        "--video",
        str(video),
        "--boxes",
        str(boxes_path),
        "--out",
        str(out),
        "--present-lead-ms",
        _num(present_lead_ms),
        "--overhead-ms",
        _num(overhead_ms),
        "--persist-passes",
        str(int(persist_passes)),
        "--shrink-alpha",
        _num(shrink_alpha),
        "--min-persist-s",
        _num(min_persist_s),
        "--max-lead-s",
        _num(max_lead_s),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout.rstrip())
    tail = "\n".join((result.stderr or "").strip().splitlines()[-5:])
    if tail:
        print(f"{warn('Warning:')} {REPLAY_BINARY_NAME}: {tail}", file=sys.stderr)
    if result.returncode != 0:
        detail = f":\n{tail}" if tail else ""
        raise RuntimeError(f"{REPLAY_BINARY_NAME} failed (code {result.returncode}){detail}")
    return result.returncode


# --- Step 3: scoring --------------------------------------------------------


def link_objects(
    dets_by_frame: dict[int, list[tuple[str, Box]]],
    gap_frames: int,
    iou_thresh: float = LINK_IOU,
) -> list[Track]:
    """Link same-class detections across frames into ground-truth objects.

    Links form greedily, highest IoU first, when the frames are at most
    ``gap_frames + 1`` apart and IoU >= ``iou_thresh``. A track never holds two
    detections in the same frame.

    Returned tracks are sorted deterministically by (first frame, class name,
    box) so equal inputs always give equal order.
    """
    nodes: list[tuple[int, str, Box]] = []
    by_frame: dict[int, list[int]] = {}
    tracks: list[Track] = []
    for frame in sorted(dets_by_frame):
        for cls, box in dets_by_frame[frame]:
            nodes.append((frame, cls, box))
            by_frame.setdefault(frame, []).append(len(nodes) - 1)
            tracks.append(Track(cls, {frame: box}))
    if not nodes:
        return []

    owner = list(range(len(tracks)))
    members: list[list[int]] = [[i] for i in range(len(tracks))]

    frames = sorted(by_frame)
    pairs: list[tuple[float, int, int]] = []
    for pos, fa in enumerate(frames):
        limit = fa + gap_frames + 1
        for next_pos in range(pos + 1, len(frames)):
            fb = frames[next_pos]
            if fb > limit:
                break
            for i in by_frame[fa]:
                _, cls_a, box_a = nodes[i]
                for j in by_frame[fb]:
                    _, cls_b, box_b = nodes[j]
                    if cls_a != cls_b:
                        continue
                    overlap = _iou(box_a, box_b)
                    if overlap >= iou_thresh:
                        pairs.append((overlap, i, j))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    for _overlap, i, j in pairs:
        ti, tj = owner[i], owner[j]
        if ti == tj:
            continue
        if any(f in tracks[ti].boxes for f in tracks[tj].boxes):
            continue
        keep, drop = (ti, tj) if len(members[ti]) >= len(members[tj]) else (tj, ti)
        tracks[keep].boxes.update(tracks[drop].boxes)
        members[keep].extend(members[drop])
        for node in members[drop]:
            owner[node] = keep
        tracks[drop] = Track(tracks[drop].cls, {})
        members[drop] = []

    ordered = [t for t in tracks if t.boxes]
    ordered.sort(
        key=lambda t: (
            min(t.boxes),
            t.cls,
            tuple(sorted((f, tuple(b)) for f, b in t.boxes.items())),
        )
    )
    return ordered


def fill_gaps(track: Track) -> Track:
    """Return a copy of ``track`` with boxes linearly interpolated on missing frames."""
    boxes = dict(track.boxes)
    frames = track.frames
    for a, b in zip(frames, frames[1:]):
        for frame in range(a + 1, b):
            t = (frame - a) / (b - a)
            boxes[frame] = tuple(
                va + (vb - va) * t for va, vb in zip(track.boxes[a], track.boxes[b])
            )
    return Track(track.cls, boxes)


def _scale_box(box: Box, sx: float, sy: float) -> Box:
    """Scale a box into live coordinates, rounding to int."""
    x, y, w, h = box
    return (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy)))


def _union_box(a: Box, b: Box) -> Box:
    """Smallest box containing both ``a`` and ``b``."""
    x0 = min(a[0], b[0])
    y0 = min(a[1], b[1])
    x1 = max(a[0] + a[2], b[0] + b[2])
    y1 = max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def _merge_class_boxes(boxes: list[Box], iou_thresh: float = GT_MERGE_IOU) -> list[Box]:
    """Merge overlapping boxes of one class/frame into their union boxes."""
    boxes = list(boxes)
    while len(boxes) > 1:
        best: float | None = None
        bi = bj = -1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                v = _iou(boxes[i], boxes[j])
                if v >= iou_thresh and (best is None or v > best):
                    best = v
                    bi, bj = i, j
        if best is None:
            break
        merged = _union_box(boxes[bi], boxes[bj])
        boxes = [b for k, b in enumerate(boxes) if k != bi and k != bj] + [merged]
    return boxes


def _merge_gt_frame(
    dets: list[tuple[str, Box]], iou_thresh: float = GT_MERGE_IOU
) -> list[tuple[str, Box]]:
    """Merge same-class boxes in one frame with IoU >= ``iou_thresh``."""
    by_cls: dict[str, list[Box]] = {}
    for cls, box in dets:
        by_cls.setdefault(cls, []).append(box)
    out: list[tuple[str, Box]] = []
    for cls in sorted(by_cls):
        for box in _merge_class_boxes(by_cls[cls], iou_thresh):
            out.append((cls, box))
    out.sort(key=lambda t: (t[0], tuple(t[1])))
    return out


def _gt_dets_by_frame(
    live_header: dict,
    live_records: list[FrameRecord],
    gt_paths: list[str | Path] | tuple[str | Path, ...] | None,
    classes: frozenset[str],
    gt_conf: float,
) -> tuple[dict[int, list[tuple[str, Box]]], list[str]]:
    """Build GT dets_by_frame, scaled to live coordinates.

    When ``gt_paths`` is empty/None, GT is the live dets filtered by
    ``gt_conf`` (legacy behavior). Otherwise GT is the union of all listed
    dets.jsonl files (live dets NOT included unless listed), each scaled by
    (live_w/gt_w, live_h/gt_h), frame-aligned by index, truncated to the min
    frame count, with same-frame/same-class duplicates (IoU >= 0.5) merged.

    Returns (dets_by_frame, gt_sources_abs). Frame-count differences <= 2
    warn; larger mismatches raise ValueError. pts always come from live dets.
    """
    if not gt_paths:
        dets_by_frame: dict[int, list[tuple[str, Box]]] = {}
        for rec in live_records:
            keep = [
                (d.class_name, d.box)
                for d in rec.dets
                if d.class_name in classes and d.confidence >= gt_conf
            ]
            if keep:
                dets_by_frame[rec.frame] = keep
        return dets_by_frame, []
    gt_sources = [str(Path(p).expanduser().resolve()) for p in gt_paths]
    live_w = int(live_header.get("width") or 0)
    live_h = int(live_header.get("height") or 0)
    if live_w <= 0 or live_h <= 0:
        raise ValueError("live dets header has no frame size — re-run bench detect")
    live_n = len(live_records)
    per_source: list[tuple[dict[int, list[tuple[str, Box]]], int]] = []
    for src in gt_sources:
        gheader, grecords = read_dets(src)
        gt_w = int(gheader.get("width") or 0)
        gt_h = int(gheader.get("height") or 0)
        if gt_w <= 0 or gt_h <= 0:
            raise ValueError(f"GT dets header has no frame size: {src}")
        gt_n = len(grecords)
        if abs(gt_n - live_n) > 2:
            raise ValueError(
                f"GT frame count mismatch: {src} has {gt_n} frames, "
                f"live has {live_n} frames (tolerance <= 2)"
            )
        sx = live_w / gt_w
        sy = live_h / gt_h
        by_frame: dict[int, list[tuple[str, Box]]] = {}
        for rec in grecords:
            scaled = [
                (d.class_name, _scale_box(d.box, sx, sy))
                for d in rec.dets
                if d.class_name in classes and d.confidence >= gt_conf
            ]
            if scaled:
                by_frame.setdefault(rec.frame, []).extend(scaled)
        per_source.append((by_frame, gt_n))
    counts = [live_n, *(n for _, n in per_source)]
    if len(set(counts)) != 1:
        print(
            f"{warn('Warning:')} GT frame count differs from live "
            f"(live={live_n}, gt={[n for _, n in per_source]}); using min",
            file=sys.stderr,
        )
    min_n = min(counts)
    live_frames = sorted(rec.frame for rec in live_records)[:min_n]
    keep_frames = set(live_frames)
    union: dict[int, list[tuple[str, Box]]] = {}
    for frame in sorted(keep_frames):
        combined: list[tuple[str, Box]] = []
        for by_frame, _ in per_source:
            combined.extend(by_frame.get(frame, []))
        if combined:
            union[frame] = _merge_gt_frame(combined)
    return union, gt_sources


def _raster(boxes, gx0: int, gy0: int, gw: int, gh: int, scale: int) -> np.ndarray:
    """Rasterize boxes onto a boolean grid of shape (gh, gw) with origin (gx0, gy0)."""
    mask = np.zeros((gh, gw), dtype=bool)
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        ax0 = max(gx0, int(math.floor(x / scale)))
        ay0 = max(gy0, int(math.floor(y / scale)))
        ax1 = min(gx0 + gw, int(math.ceil((x + w) / scale)))
        ay1 = min(gy0 + gh, int(math.ceil((y + h) / scale)))
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        mask[ay0 - gy0 : ay1 - gy0, ax0 - gx0 : ax1 - gx0] = True
    return mask


def coverage_fraction(box: Box, others, scale: int = GRID) -> float:
    """Fraction of ``box`` covered by the union of ``others``.

    Rasterized on a grid downscaled by ``scale`` (exact enough, fast with numpy).
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return 0.0
    gx0 = int(math.floor(x / scale))
    gy0 = int(math.floor(y / scale))
    gx1 = max(gx0 + 1, int(math.ceil((x + w) / scale)))
    gy1 = max(gy0 + 1, int(math.ceil((y + h) / scale)))
    mask = _raster(others, gx0, gy0, gx1 - gx0, gy1 - gy0, scale)
    return float(np.count_nonzero(mask)) / float(mask.size)


def dilate_box(box: Box, factor: float = GT_DILATE) -> Box:
    """Expand a box by ``factor`` of its own size on every side (centered)."""
    x, y, w, h = box
    return (x - w * factor, y - h * factor, w * (1 + 2 * factor), h * (1 + 2 * factor))


def excess_fraction(displayed, gt_boxes, width: int, height: int, scale: int = GRID) -> float:
    """Fraction of the frame covered by displayed boxes but not by the GT boxes."""
    if width <= 0 or height <= 0:
        return 0.0
    gw = max(1, math.ceil(width / scale))
    gh = max(1, math.ceil(height / scale))
    shown = _raster(displayed, 0, 0, gw, gh, scale)
    truth = _raster(gt_boxes, 0, 0, gw, gh, scale)
    return float(np.count_nonzero(shown & ~truth)) / float(gw * gh)


def nearest_display(display_pts: list[float], pts: float) -> int:
    """Index of the display line whose display_pts is nearest to ``pts``."""
    i = bisect_left(display_pts, pts)
    if i <= 0:
        return 0
    if i >= len(display_pts):
        return len(display_pts) - 1
    if (pts - display_pts[i - 1]) <= (display_pts[i] - pts):
        return i - 1
    return i


def on_screen_display(display_pts: list[float], pts: float) -> int:
    """Index of the displayed line that is on screen at time ``pts``.

    The latest line with ``display_pts <= pts``; clamps to the first line
    when ``pts`` predates every display line.
    """
    if not display_pts:
        return 0
    i = bisect_right(display_pts, pts) - 1
    if i < 0:
        return 0
    if i >= len(display_pts):
        return len(display_pts) - 1
    return i


def count_flickers(covered: list[bool]) -> int:
    """Count covered -> uncovered transitions after the first covered frame."""
    flickers = 0
    seen = False
    was_covered = False
    for is_covered in covered:
        if is_covered:
            seen = True
            was_covered = True
        elif seen and was_covered:
            flickers += 1
            was_covered = False
    return flickers


def over_censor_pct(
    gt_by_frame: dict[int, list[Box]],
    disp_pts: list[float],
    disp_boxes: list[tuple[Box, ...]],
    width: int,
    height: int,
    pts_of,
    dets_frames=None,
    scale: int = GRID,
) -> float:
    """Percent of the frame over-censored: mean over EVERY displayed frame.

    Each displayed line is compared against the dilated GT of the dets frame
    nearest to its ``display_pts`` (empty when that frame has no GT, so any
    displayed box counts fully as excess). This charges boxes that persist
    after content is gone instead of dropping those frames.
    """
    if not disp_pts or width <= 0 or height <= 0:
        return 0.0
    if dets_frames is None:
        dets_frames = sorted(gt_by_frame)
    else:
        dets_frames = sorted(dets_frames)
    frame_pts = sorted((pts_of(f), f) for f in dets_frames)
    dets_pts_sorted = [p for p, _ in frame_pts]
    dets_order = [f for _, f in frame_pts]
    total = 0.0
    for idx, dpts in enumerate(disp_pts):
        boxes = disp_boxes[idx]
        gt_boxes: list[Box] = []
        if dets_pts_sorted:
            j = nearest_display(dets_pts_sorted, dpts)
            gt_boxes = gt_by_frame.get(dets_order[j], [])
        dilated = [dilate_box(b) for b in gt_boxes]
        total += excess_fraction(boxes, dilated, width, height, scale)
    return 100.0 * total / len(disp_pts)


def _p90(values: list[float]) -> float:
    s = sorted(values)
    idx = max(0, math.ceil(0.9 * len(s)) - 1)
    return s[min(idx, len(s) - 1)]


def _group_metrics(
    tracks: list[Track],
    disp_pts: list[float],
    disp_boxes: list[tuple[Box, ...]],
    width: int,
    height: int,
    pts_of,
    minutes: float,
    fps: float = DEFAULT_FPS,
    dets_frames=None,
    include_over_censor: bool = True,
    live_by_frame: dict[int, list[tuple[str, Box]]] | None = None,
) -> dict:
    """Score one group of ground-truth objects (all objects, or one class).

    Displayed boxes carry no class, so ``over_censor_pct`` is only meaningful
    for the overall group (dilated GT of all censored classes); per-class
    groups report ``over_censor_pct`` as None.
    """
    leaks: list[float] = []
    flicker_total = 0
    delays: list[float] = []
    gt_by_frame: dict[int, list[Box]] = {}
    n_gt_boxes = 0
    n_missed = 0
    n_missed_by_live = 0
    frame_s = 1.0 / fps if math.isfinite(fps) and fps > 0 else 0.0
    for track in tracks:
        frames = track.frames
        n_gt_boxes += len(frames)
        first_pts = pts_of(frames[0])
        last_pts = pts_of(frames[-1])
        flags: list[bool] = []
        first_cover: float | None = None
        for frame in frames:
            gt_by_frame.setdefault(frame, []).append(track.boxes[frame])
            boxes: tuple[Box, ...] = ()
            if disp_pts:
                boxes = disp_boxes[on_screen_display(disp_pts, pts_of(frame))]
            covered_frac = coverage_fraction(track.boxes[frame], boxes)
            leaks.append(1.0 - covered_frac)
            covered = covered_frac >= COVER_THRESHOLD
            flags.append(covered)
            if covered and first_cover is None:
                first_cover = (pts_of(frame) - first_pts) * 1000.0
        flicker_total += count_flickers(flags)
        if first_cover is not None:
            delays.append(first_cover)
        else:
            delays.append((last_pts - first_pts + frame_s) * 1000.0)
        if not any(flags):
            n_missed += 1
        if live_by_frame is not None:
            seen_by_live = False
            for frame in frames:
                for cls_live, box_live in live_by_frame.get(frame, ()):
                    if cls_live != track.cls:
                        continue
                    if _iou(track.boxes[frame], box_live) >= LIVE_MATCH_IOU:
                        seen_by_live = True
                        break
                if seen_by_live:
                    break
            if not seen_by_live:
                n_missed_by_live += 1

    n_leak_frames = sum(1 for leak in leaks if leak > 1.0 - COVER_THRESHOLD)
    if include_over_censor:
        over = over_censor_pct(
            gt_by_frame, disp_pts, disp_boxes, width, height, pts_of, dets_frames
        )
    else:
        over = None
    return {
        "leak_frac": fmean(leaks) if leaks else 0.0,
        "leak_frames_pct": 100.0 * n_leak_frames / len(leaks) if leaks else 0.0,
        "flicker_per_min": flicker_total / minutes if minutes > 0 else 0.0,
        "first_cover_ms": float(median(delays)) if delays else None,
        "first_cover_p90_ms": float(_p90(delays)) if delays else None,
        "over_censor_pct": over,
        "n_objects": len(tracks),
        "n_gt_boxes": n_gt_boxes,
        "objects_missed_pct": 100.0 * n_missed / len(tracks) if tracks else 0.0,
        "objects_missed_by_live_pct": 100.0 * n_missed_by_live / len(tracks) if tracks else 0.0,
    }


def score(
    dets_path: str | Path,
    displayed_path: str | Path,
    classes: frozenset[str],
    gt_conf: float = DEFAULT_GT_CONF,
    gap_s: float = DEFAULT_GAP_S,
    gt_dets: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict:
    """Score replayed boxes against oracle ground truth.

    Ground truth is the per-frame detections of ``classes`` with confidence >=
    ``gt_conf``, unpadded, linked into objects (``link_objects``) and gap-filled
    (``fill_gaps``). By default GT is ``dets_path``; with ``gt_dets`` it is the
    union of the listed dets.jsonl files scaled to the live resolution (the
    live file is NOT included unless listed). GT frames match the displayed
    line that is on screen at their pts (latest ``display_pts`` <= pts).
    Over-censor averages over every displayed frame against the dilated GT of
    the nearest dets frame, so persistence after content is gone is charged;
    per-class ``over_censor_pct`` is None (displayed boxes carry no class).
    Returns ``{"overall": {...}, "per_class": {...}}``; the run's config is
    added by :func:`cmd_replay`.
    """
    classes = frozenset(classes)
    header, records = read_dets(dets_path)
    _summary, displayed = read_displayed(displayed_path)
    displayed = sorted(displayed, key=lambda d: d.display_pts)

    fps = float(header.get("fps") or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = DEFAULT_FPS
    n_frames = int(header.get("frames") or len(records) or 0)
    minutes = n_frames / fps / 60.0
    gap_frames = max(1, round(gap_s * fps))
    width = int(header.get("width") or 0)
    height = int(header.get("height") or 0)

    pts_by_frame = {rec.frame: rec.pts for rec in records}

    def pts_of(frame: int) -> float:
        return pts_by_frame.get(frame, frame / fps)

    dets_by_frame, _gt_sources = _gt_dets_by_frame(header, records, gt_dets, classes, gt_conf)
    live_by_frame: dict[int, list[tuple[str, Box]]] = {}
    for rec in records:
        keep = [(d.class_name, d.box) for d in rec.dets if d.class_name in classes]
        if keep:
            live_by_frame[rec.frame] = keep

    tracks = [fill_gaps(t) for t in link_objects(dets_by_frame, gap_frames)]
    disp_pts = [d.display_pts for d in displayed]
    disp_boxes = [d.boxes for d in displayed]
    dets_frames = [rec.frame for rec in records]

    def group(items: list[Track], include_over: bool) -> dict:
        return _group_metrics(
            items,
            disp_pts,
            disp_boxes,
            width,
            height,
            pts_of,
            minutes,
            fps,
            dets_frames,
            include_over,
            live_by_frame,
        )

    per_class: dict[str, dict] = {}
    for cls in sorted({t.cls for t in tracks}):
        per_class[cls] = group([t for t in tracks if t.cls == cls], False)
    return {"overall": group(tracks, True), "per_class": per_class}


# --- Sweep ------------------------------------------------------------------


def config_name(
    padding: float,
    persist_passes: int,
    min_persist_s: float,
    shrink_alpha: float,
    min_padding: int = 0,
) -> str:
    """Deterministic run subdirectory name for a replay config."""
    pad = _num(padding)
    mp = _num(min_persist_s)
    sa = _num(shrink_alpha)
    name = f"pad{pad}-pp{int(persist_passes)}-mp{mp}-sa{sa}"
    if min_padding:
        name += f"-mpx{int(min_padding)}"
    return name


def _fmt_ms(value) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}"


def print_table(results: list[dict]) -> None:
    """Print a compact aligned results table, best (lowest leak_frac) first."""
    columns = ("config", "leak_frac", "leak_frames%", "flick/min", "first_cover_ms", "over_censor%")
    rows = [tuple(str(c) for c in columns)]
    for entry in results:
        o = entry["overall"]
        rows.append(
            (
                entry["name"],
                f"{o['leak_frac']:.3f}",
                f"{o['leak_frames_pct']:.1f}",
                f"{o['flicker_per_min']:.1f}",
                _fmt_ms(o["first_cover_ms"]),
                f"{o['over_censor_pct']:.2f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True)))
    print(dim("─" * (sum(widths) + 2 * (len(widths) - 1))))
    for row in rows[1:]:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))


def _sweep_configs(censor: str, classes: frozenset[str], min_padding: int = 0) -> list[dict]:
    configs = []
    for padding in SWEEP_PADDINGS:
        for persist_passes, min_persist_s in SWEEP_PERSIST:
            for shrink_alpha in SWEEP_SHRINK:
                configs.append(
                    {
                        "name": config_name(
                            padding, persist_passes, min_persist_s, shrink_alpha, min_padding
                        ),
                        "config": {
                            "censor": censor,
                            "classes": sorted(classes),
                            "padding": padding,
                            "min_padding": min_padding,
                            "persist_passes": persist_passes,
                            "min_persist_s": min_persist_s,
                            "shrink_alpha": shrink_alpha,
                            "present_lead_ms": DEFAULT_PRESENT_LEAD_MS,
                            "overhead_ms": DEFAULT_OVERHEAD_MS,
                        },
                    }
                )
    return configs


def sweep(
    det_dir: str | Path,
    censor: str = DEFAULT_CENSOR,
    *,
    gt_conf: float = DEFAULT_GT_CONF,
    gap_s: float = DEFAULT_GAP_S,
    present_lead_ms: float = DEFAULT_PRESENT_LEAD_MS,
    overhead_ms: float = DEFAULT_OVERHEAD_MS,
    min_padding: int = 0,
    gt_dets: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict:
    """Run a grid of replay configs over one detect directory; write sweep.json."""
    base = Path(det_dir).expanduser()
    dets_path = require_dets(base)
    header, _records = read_dets(dets_path)
    video = require_video(header, dets_path)
    classes = censor_classes(censor)
    gt_sources = [str(Path(p).expanduser().resolve()) for p in (gt_dets or [])]

    results: list[dict] = []
    for entry in _sweep_configs(censor, classes, min_padding):
        cfg = entry["config"]
        cfg["gt_sources"] = list(gt_sources)
        cfg["gt_conf"] = gt_conf
        name = entry["name"]
        run_dir = base / name
        run_dir.mkdir(parents=True, exist_ok=True)
        boxes_path = write_boxes(
            dets_path, run_dir / BOXES_NAME, classes, cfg["padding"], cfg["min_padding"]
        )
        run_replay(
            video,
            boxes_path,
            run_dir / DISPLAYED_NAME,
            present_lead_ms=present_lead_ms,
            overhead_ms=overhead_ms,
            persist_passes=cfg["persist_passes"],
            min_persist_s=cfg["min_persist_s"],
            shrink_alpha=cfg["shrink_alpha"],
        )
        scored = score(dets_path, run_dir / DISPLAYED_NAME, classes, gt_conf, gap_s, gt_dets)
        payload = {"config": cfg, **scored}
        with open(run_dir / SCORE_NAME, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2) + "\n")
        results.append({"name": name, **payload})
        print(dim(f"  {name} done"))

    results.sort(key=lambda e: e["overall"]["leak_frac"])
    sweep_path = base / SWEEP_NAME
    payload = {
        "video": video,
        "censor": censor,
        "gt_conf": gt_conf,
        "gap_s": gap_s,
        "gt_sources": list(gt_sources),
        "present_lead_ms": present_lead_ms,
        "overhead_ms": overhead_ms,
        "results": results,
    }
    with open(sweep_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2) + "\n")
    print_table(results)
    print(f"{success('Saved')} {bold(str(sweep_path))}")
    return payload


# --- CLI entry points -------------------------------------------------------


def _fail(exc: Exception) -> None:
    print(f"{error('Error:')} {exc}", file=sys.stderr)
    sys.exit(1)


def cmd_detect(args) -> None:
    """bsafe bench detect"""
    try:
        detect(
            args.input,
            args.output,
            model=args.model,
            extra_scales=getattr(args, "extra_scales", None),
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        _fail(e)


def cmd_replay(args) -> None:
    """bsafe bench replay"""
    try:
        base = Path(args.dir).expanduser()
        dets_path = require_dets(base)
        header, records = read_dets(dets_path)
        video = require_video(header, dets_path)
        classes = censor_classes(args.censor)
        if args.padding < 0:
            raise ValueError("--padding must be >= 0")
        if getattr(args, "min_padding", 0) < 0:
            raise ValueError("--min-padding must be >= 0")
        min_padding = int(getattr(args, "min_padding", 0))
        name = args.name or config_name(
            args.padding, args.persist_passes, args.min_persist_s, args.shrink_alpha, min_padding
        )
        run_dir = base / name
        run_dir.mkdir(parents=True, exist_ok=True)
        gt_dets = list(getattr(args, "gt_dets", None) or [])
        gt_conf = float(getattr(args, "gt_conf", DEFAULT_GT_CONF))
        gt_sources = [str(Path(p).expanduser().resolve()) for p in gt_dets]
        cfg = {
            "censor": args.censor,
            "classes": sorted(classes),
            "padding": args.padding,
            "min_padding": min_padding,
            "persist_passes": args.persist_passes,
            "min_persist_s": args.min_persist_s,
            "shrink_alpha": args.shrink_alpha,
            "present_lead_ms": args.present_lead_ms,
            "overhead_ms": args.overhead_ms,
            "max_lead_s": DEFAULT_MAX_LEAD_S,
            "gt_sources": list(gt_sources),
            "gt_conf": gt_conf,
        }
        print(
            f"{dim('Config:')} censor={args.censor}, padding={_num(args.padding)}, "
            f"min_padding={min_padding}, "
            f"persist_passes={args.persist_passes}, min_persist_s={_num(args.min_persist_s)}, "
            f"shrink_alpha={_num(args.shrink_alpha)}, present_lead_ms={_num(args.present_lead_ms)}, "
            f"overhead_ms={_num(args.overhead_ms)}"
        )
        boxes_path = write_boxes(
            dets_path, run_dir / BOXES_NAME, classes, args.padding, min_padding
        )
        print(f"Boxes: {bold(str(boxes_path))} {dim(f'({len(records)} frames)')}")
        run_replay(
            video,
            boxes_path,
            run_dir / DISPLAYED_NAME,
            present_lead_ms=args.present_lead_ms,
            overhead_ms=args.overhead_ms,
            persist_passes=args.persist_passes,
            min_persist_s=args.min_persist_s,
            shrink_alpha=args.shrink_alpha,
        )
        print(f"Displayed: {bold(str(run_dir / DISPLAYED_NAME))}")
        payload = {
            "config": cfg,
            **score(
                dets_path,
                run_dir / DISPLAYED_NAME,
                classes,
                gt_conf,
                DEFAULT_GAP_S,
                gt_dets or None,
            ),
        }
        score_path = run_dir / SCORE_NAME
        with open(score_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2) + "\n")
        print_score(payload["overall"])
        print(f"{success('Saved')} {bold(str(score_path))}")
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        _fail(e)


def print_score(overall: dict) -> None:
    """Print a one-line score summary with the headline metrics."""
    leak = f"{overall['leak_frac']:.3f}"
    leak_frames = f"{overall['leak_frames_pct']:.1f}%"
    flicker = f"{overall['flicker_per_min']:.1f}/min"
    over_censor = f"{overall['over_censor_pct']:.2f}%"
    print(
        f"leak_frac {info(leak)}  leak_frames {info(leak_frames)}  "
        f"flicker {info(flicker)}  first_cover {info(_fmt_ms(overall['first_cover_ms']))} ms  "
        f"over_censor {info(over_censor)}"
    )
    counts = f"objects {overall['n_objects']}, gt_boxes {overall['n_gt_boxes']}"
    print(dim(counts))
    missed = float(overall.get("objects_missed_pct", 0.0))
    missed_live = float(overall.get("objects_missed_by_live_pct", 0.0))
    print(
        f"objects_missed {info(f'{missed:.1f}%')}  "
        f"objects_missed_by_live {info(f'{missed_live:.1f}%')}"
    )


def cmd_sweep(args) -> None:
    """bsafe bench sweep"""
    try:
        sweep(
            args.dir,
            args.censor,
            min_padding=int(getattr(args, "min_padding", 0)),
            gt_conf=float(getattr(args, "gt_conf", DEFAULT_GT_CONF)),
            gt_dets=list(getattr(args, "gt_dets", None) or []) or None,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        _fail(e)
