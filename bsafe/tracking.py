"""Temporal smoothing and box tracking to reduce flicker."""


class _TrackedBox:
    """Internal state for a single tracked box."""

    __slots__ = ("x", "y", "w", "h", "frames_missing")

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frames_missing = 0

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (round(self.x), round(self.y), round(self.w), round(self.h))


def _iou(a: _TrackedBox, bx: float, by: float, bw: float, bh: float) -> float:
    """Compute IoU between a tracked box and a raw (x, y, w, h)."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax1, bx)
    iy1 = max(ay1, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = a.w * a.h
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class BoxTracker:
    """Per-display temporal smoothing and persistence for censor boxes.

    Args:
        persist_frames: How many frames a box stays visible after it disappears.
        smooth_alpha: EMA weight for new positions (1.0 = no smoothing, 0.0 = frozen).
        iou_threshold: Minimum IoU to match a new box to an existing track.
    """

    def __init__(
        self,
        persist_frames: int = 8,
        smooth_alpha: float = 0.5,
        iou_threshold: float = 0.2,
    ):
        self.persist_frames = persist_frames
        self.smooth_alpha = smooth_alpha
        self.iou_threshold = iou_threshold
        # Per-display tracked boxes
        self._tracks: dict[int, list[_TrackedBox]] = {}

    def update(
        self,
        display_id: int,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Update tracks for a display and return smoothed boxes to render."""
        tracks = self._tracks.get(display_id, [])

        # Greedy IoU-based matching
        matched_track_indices: set[int] = set()
        matched_box_indices: set[int] = set()

        # Build (score, track_idx, box_idx) pairs sorted by IoU descending
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(tracks):
            for bi, (bx, by, bw, bh) in enumerate(boxes):
                score = _iou(track, float(bx), float(by), float(bw), float(bh))
                if score >= self.iou_threshold:
                    pairs.append((score, ti, bi))
        pairs.sort(reverse=True)

        alpha = self.smooth_alpha
        for score, ti, bi in pairs:
            if ti in matched_track_indices or bi in matched_box_indices:
                continue
            matched_track_indices.add(ti)
            matched_box_indices.add(bi)
            # EMA smooth the position
            track = tracks[ti]
            bx, by, bw, bh = boxes[bi]
            track.x = alpha * bx + (1 - alpha) * track.x
            track.y = alpha * by + (1 - alpha) * track.y
            track.w = alpha * bw + (1 - alpha) * track.w
            track.h = alpha * bh + (1 - alpha) * track.h
            track.frames_missing = 0

        # Unmatched tracks: increment missing counter
        for ti, track in enumerate(tracks):
            if ti not in matched_track_indices:
                track.frames_missing += 1

        # Unmatched boxes: create new tracks
        for bi, (bx, by, bw, bh) in enumerate(boxes):
            if bi not in matched_box_indices:
                tracks.append(_TrackedBox(float(bx), float(by), float(bw), float(bh)))

        # Prune expired tracks
        tracks = [t for t in tracks if t.frames_missing < self.persist_frames]

        self._tracks[display_id] = tracks

        return [t.to_tuple() for t in tracks]

    def clear(self, display_id: int) -> None:
        """Drop all tracks for a display (used when a stale result is rejected)."""
        self._tracks.pop(display_id, None)
