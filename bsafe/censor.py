"""Censor filtering and box padding for NSFW detections."""

from dataclasses import dataclass

from bsafe.detector import Detection

# How much the full-censor mode expands the area; 3x means width and height
# are each tripled (9x area), centered on the original box.
FULL_CENSOR_MULTIPLIER = 3

# BUTTOCKS_EXPOSED is intentionally excluded — too many false positives in practice
# (e.g. tight clothing, seated posture) and low user-reported value for censoring. Opt in with --buttocks.
CENSOR_PRESETS: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "female": frozenset(
        {
            "FEMALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
        }
    ),
    "male": frozenset(
        {
            "MALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
        }
    ),
    "all": frozenset(
        {
            "FEMALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
        }
    ),
    # body: every exposed body-part class except faces. BELLY/ARMPITS/FEET_COVERED stay out:
    # they fire on ordinary clothing.
    "body": frozenset(
        {
            "FEMALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
            "BUTTOCKS_EXPOSED",
            "MALE_BREAST_EXPOSED",
            "FEET_EXPOSED",
            "BELLY_EXPOSED",
            "ARMPITS_EXPOSED",
        }
    ),
}


@dataclass(frozen=True)
class CensorConfig:
    """Groups all censor-related options into a single value object."""

    preset: str = "all"
    covered: bool = False
    face_male: bool = False
    face_female: bool = False
    feet: bool = False
    buttocks: bool = False

    def resolve_classes(self) -> frozenset[str]:
        """Build the set of classes to censor from preset plus additive flags."""
        return resolve_censor_classes(
            self.preset,
            covered=self.covered,
            face_male=self.face_male,
            face_female=self.face_female,
            feet=self.feet,
            buttocks=self.buttocks,
        )


def resolve_censor_classes(
    preset: str,
    *,
    covered: bool = False,
    face_male: bool = False,
    face_female: bool = False,
    feet: bool = False,
    buttocks: bool = False,
) -> frozenset[str]:
    """Build the set of classes to censor from a preset plus additive flags."""
    if preset not in CENSOR_PRESETS:
        raise ValueError(
            f"unknown censor preset '{preset}'. Known presets: {', '.join(sorted(CENSOR_PRESETS))}"
        )
    classes = set(CENSOR_PRESETS[preset])
    if covered and preset != "none":
        classes.update({"ANUS_COVERED", "BUTTOCKS_COVERED"})
        # Male breast is considered SFW in most cultures, so only add female breast covered.
        if preset in ("female", "all", "body"):
            classes.update({"FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED"})
    if face_male:
        classes.add("FACE_MALE")
    if face_female:
        classes.add("FACE_FEMALE")
    if feet:
        classes.add("FEET_EXPOSED")
    if buttocks:
        classes.add("BUTTOCKS_EXPOSED")
    return frozenset(classes)


def filter_detections(
    detections: list[Detection],
    classes: frozenset[str] = CENSOR_PRESETS["all"],
) -> list[Detection]:
    """Return only detections whose class_name is in the given set."""
    return [d for d in detections if d.class_name in classes]


def pad_box(
    box: tuple[int, int, int, int],
    padding: float,
    display_w: int,
    display_h: int,
    min_padding: int = 0,
) -> tuple[int, int, int, int]:
    """Expand (x, y, w, h) by padding fraction, clamp to bounds, return (x, y, w, h)."""
    x, y, w, h = box
    pad_x = max(int(w * padding), min_padding)
    pad_y = max(int(h * padding), min_padding)

    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y)
    nx2 = min(display_w, x + w + pad_x)
    ny2 = min(display_h, y + h + pad_y)

    return (nx, ny, nx2 - nx, ny2 - ny)


def merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping (x, y, w, h) boxes into their bounding unions.

    Iterates until no more merges occur. O(n^2) per pass but n is single-digit.
    """
    if len(boxes) <= 1:
        return list(boxes)

    # Convert to (x1, y1, x2, y2)
    rects = [(x, y, x + w, y + h) for x, y, w, h in boxes]

    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(rects)

        for i in range(len(rects)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = rects[j]
                # Check intersection
                if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used[j] = True
                    changed = True
            merged.append((ax1, ay1, ax2, ay2))

        rects = merged

    # Convert back to (x, y, w, h)
    return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in rects]


def expand_boxes(
    boxes: list[tuple[int, int, int, int]],
    multiplier: float,
    display_w: int,
    display_h: int,
) -> list[tuple[int, int, int, int]]:
    """Scale each box's width/height by multiplier, centered on the original box center, clamped to display bounds."""
    result = []
    for x, y, w, h in boxes:
        cx = x + w / 2
        cy = y + h / 2
        nw = w * multiplier
        nh = h * multiplier
        nx = int(max(0, cx - nw / 2))
        ny = int(max(0, cy - nh / 2))
        nx2 = int(min(display_w, cx + nw / 2))
        ny2 = int(min(display_h, cy + nh / 2))
        result.append((nx, ny, nx2 - nx, ny2 - ny))
    return result


def build_censor_boxes(
    detections: list[Detection],
    classes: frozenset[str],
    padding: float,
    display_w: int,
    display_h: int,
    min_padding: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Filter detections, pad boxes, merge overlaps, return list of (x, y, w, h) tuples."""
    filtered = filter_detections(detections, classes)
    padded = [pad_box(d.box, padding, display_w, display_h, min_padding) for d in filtered]
    return merge_overlapping_boxes(padded)
