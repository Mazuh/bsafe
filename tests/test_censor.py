"""Tests for censor filtering and box padding."""

import pytest

from bsafe.censor import (
    CENSOR_PRESETS,
    FULL_CENSOR_MULTIPLIER,
    CensorConfig,
    build_censor_boxes,
    expand_boxes,
    filter_detections,
    merge_overlapping_boxes,
    pad_box,
    resolve_censor_classes,
)
from bsafe.detector import Detection


def _det(class_name, box=(100, 100, 80, 60), confidence=0.9):
    return Detection(class_name=class_name, confidence=confidence, box=box)


def test_filter_detections_keeps_matching():
    dets = [
        _det("FEMALE_BREAST_EXPOSED"),
        _det("FACE_FEMALE"),
        _det("MALE_GENITALIA_EXPOSED"),
    ]
    result = filter_detections(dets, CENSOR_PRESETS["all"])
    assert len(result) == 2
    assert result[0].class_name == "FEMALE_BREAST_EXPOSED"
    assert result[1].class_name == "MALE_GENITALIA_EXPOSED"


def test_filter_detections_empty_input():
    assert filter_detections([], CENSOR_PRESETS["all"]) == []


def test_filter_detections_none_match():
    dets = [_det("FACE_FEMALE"), _det("BELLY_EXPOSED")]
    assert filter_detections(dets, CENSOR_PRESETS["all"]) == []


def test_pad_box_no_padding():
    # (x=100, y=100, w=80, h=60) with 0 padding → same box
    assert pad_box((100, 100, 80, 60), 0.0, 1920, 1080) == (100, 100, 80, 60)


def test_pad_box_with_padding():
    # box (x=100, y=100, w=100, h=100), padding 0.2 → expand 20px each side
    x, y, w, h = pad_box((100, 100, 100, 100), 0.2, 1920, 1080)
    assert x == 80
    assert y == 80
    assert w == 140
    assert h == 140


def test_pad_box_clamps_to_bounds():
    # Box near top-left corner (x=5, y=5, w=40, h=40), large padding
    x, y, w, h = pad_box((5, 5, 40, 40), 0.5, 100, 100)
    assert x == 0
    assert y == 0
    assert w >= 40  # at least original width
    assert h >= 40


def test_pad_box_clamps_to_bottom_right():
    # Box near bottom-right, padding would exceed display
    # (x=950, y=550, w=40, h=40) in 1000x600 display, padding 0.5 → 20px each side
    x, y, w, h = pad_box((950, 550, 40, 40), 0.5, 1000, 600)
    assert x + w <= 1000
    assert y + h <= 600


def test_build_censor_boxes():
    # BUTTOCKS_EXPOSED is intentionally not censored (see censor.py)
    dets = [
        _det("FEMALE_BREAST_EXPOSED", box=(100, 100, 80, 60)),
        _det("FACE_FEMALE", box=(300, 300, 80, 60)),
        _det("BUTTOCKS_EXPOSED", box=(500, 500, 80, 60)),
    ]
    boxes = build_censor_boxes(dets, CENSOR_PRESETS["all"], 0.0, 1920, 1080)
    assert len(boxes) == 1
    assert boxes[0] == (100, 100, 80, 60)


def test_build_censor_boxes_empty():
    assert build_censor_boxes([], CENSOR_PRESETS["all"], 0.2, 1920, 1080) == []


def test_pad_box_min_padding_grows_small_box():
    # 20x20 box with padding 0.0 would stay put; floor of 48 px per side wins.
    x, y, w, h = pad_box((100, 100, 20, 20), 0.0, 1920, 1080, min_padding=48)
    assert (x, y, w, h) == (52, 52, 116, 116)


def test_pad_box_min_padding_leaves_large_box_untouched():
    # 200x200 box with padding 0.4 → 80 px per side, above the 48 px floor.
    assert pad_box((500, 500, 200, 200), 0.4, 1920, 1080, min_padding=48) == pad_box(
        (500, 500, 200, 200), 0.4, 1920, 1080
    )
    assert pad_box((500, 500, 200, 200), 0.4, 1920, 1080, min_padding=48) == (
        420,
        420,
        360,
        360,
    )


def test_pad_box_min_padding_zero_is_legacy():
    box = (100, 100, 80, 60)
    assert pad_box(box, 0.2, 1920, 1080, min_padding=0) == pad_box(box, 0.2, 1920, 1080)
    assert pad_box(box, 0.0, 1920, 1080, min_padding=0) == box


def test_pad_box_min_padding_clamps_to_edges():
    # Near top-left: floor would push past 0, must clamp like today.
    x, y, w, h = pad_box((5, 5, 20, 20), 0.0, 200, 200, min_padding=48)
    assert x == 0
    assert y == 0
    assert x + w <= 200
    assert y + h <= 200
    assert w == 73  # 5 + 20 + 48 clamped at 0
    assert h == 73


def test_build_censor_boxes_min_padding_passthrough():
    dets = [_det("FEMALE_BREAST_EXPOSED", box=(100, 100, 20, 20))]
    boxes = build_censor_boxes(dets, CENSOR_PRESETS["all"], 0.0, 1920, 1080, min_padding=48)
    assert boxes == [(52, 52, 116, 116)]
    # Default keeps legacy output.
    legacy = build_censor_boxes(dets, CENSOR_PRESETS["all"], 0.0, 1920, 1080)
    assert legacy == [(100, 100, 20, 20)]


# --- merge_overlapping_boxes tests ---


def test_merge_non_overlapping_stay_separate():
    boxes = [(0, 0, 50, 50), (200, 200, 50, 50)]
    result = merge_overlapping_boxes(boxes)
    assert len(result) == 2
    assert (0, 0, 50, 50) in result
    assert (200, 200, 50, 50) in result


def test_merge_overlapping_pair():
    boxes = [(0, 0, 60, 60), (30, 30, 60, 60)]
    result = merge_overlapping_boxes(boxes)
    assert len(result) == 1
    assert result[0] == (0, 0, 90, 90)


def test_merge_transitive_overlap():
    # A overlaps B, B overlaps C → all three merge into one
    boxes = [(0, 0, 40, 40), (30, 0, 40, 40), (60, 0, 40, 40)]
    result = merge_overlapping_boxes(boxes)
    assert len(result) == 1
    assert result[0] == (0, 0, 100, 40)


def test_merge_single_box_unchanged():
    boxes = [(10, 20, 30, 40)]
    assert merge_overlapping_boxes(boxes) == [(10, 20, 30, 40)]


def test_merge_empty():
    assert merge_overlapping_boxes([]) == []


# --- expand_boxes tests ---


def test_expand_boxes_basic():
    # Box (100, 100, 100, 100) with 3x multiplier → centered expansion
    # Center: (150, 150), new size: 300x300 → new box: (0, 0, 300, 300)
    boxes = [(100, 100, 100, 100)]
    result = expand_boxes(boxes, 3, 1920, 1080)
    assert len(result) == 1
    assert result[0] == (0, 0, 300, 300)


def test_expand_boxes_no_expansion():
    boxes = [(100, 100, 50, 50)]
    result = expand_boxes(boxes, 1, 1920, 1080)
    assert result == [(100, 100, 50, 50)]


def test_expand_boxes_clamps_to_bounds():
    # Box near edge: center at (25, 25), 3x of 50 = 150 → would go to (-50, -50) but clamped
    boxes = [(0, 0, 50, 50)]
    result = expand_boxes(boxes, 3, 200, 200)
    x, y, w, h = result[0]
    assert x >= 0
    assert y >= 0
    assert x + w <= 200
    assert y + h <= 200


def test_expand_boxes_clamps_bottom_right():
    # Box near bottom-right corner
    boxes = [(180, 180, 20, 20)]
    result = expand_boxes(boxes, 3, 200, 200)
    x, y, w, h = result[0]
    assert x >= 0
    assert y >= 0
    assert x + w <= 200
    assert y + h <= 200


def test_expand_boxes_empty():
    assert expand_boxes([], 3, 1920, 1080) == []


def test_expand_boxes_multiple():
    boxes = [(100, 100, 50, 50), (500, 500, 50, 50)]
    result = expand_boxes(boxes, 2, 1920, 1080)
    assert len(result) == 2


def test_full_censor_multiplier_value():
    assert FULL_CENSOR_MULTIPLIER == 3


# --- resolve_censor_classes tests ---


def test_resolve_no_extra_flags_matches_preset():
    for preset in ("none", "female", "male", "all", "body"):
        assert resolve_censor_classes(preset) == CENSOR_PRESETS[preset]


def test_resolve_none_returns_empty():
    assert resolve_censor_classes("none") == frozenset()


def test_resolve_none_with_face_flags():
    result = resolve_censor_classes("none", face_male=True)
    assert result == frozenset({"FACE_MALE"})

    result = resolve_censor_classes("none", face_female=True)
    assert result == frozenset({"FACE_FEMALE"})


def test_resolve_none_with_feet():
    result = resolve_censor_classes("none", feet=True)
    assert result == frozenset({"FEET_EXPOSED"})


def test_resolve_none_covered_ignored():
    result = resolve_censor_classes("none", covered=True)
    assert result == frozenset()


def test_resolve_none_combined_additive_flags():
    result = resolve_censor_classes("none", face_male=True, face_female=True, feet=True)
    assert result == frozenset({"FACE_MALE", "FACE_FEMALE", "FEET_EXPOSED"})


def test_resolve_covered_female():
    result = resolve_censor_classes("female", covered=True)
    assert "ANUS_COVERED" in result
    assert "BUTTOCKS_COVERED" in result
    assert "FEMALE_BREAST_COVERED" in result
    assert "FEMALE_GENITALIA_COVERED" in result
    # Original classes still present
    assert "FEMALE_BREAST_EXPOSED" in result


def test_resolve_covered_male():
    result = resolve_censor_classes("male", covered=True)
    assert "ANUS_COVERED" in result
    assert "BUTTOCKS_COVERED" in result
    assert "FEMALE_BREAST_COVERED" not in result
    assert "FEMALE_GENITALIA_COVERED" not in result


def test_resolve_covered_all():
    result = resolve_censor_classes("all", covered=True)
    assert "ANUS_COVERED" in result
    assert "BUTTOCKS_COVERED" in result
    assert "FEMALE_BREAST_COVERED" in result
    assert "FEMALE_GENITALIA_COVERED" in result


def test_body_preset_contents():
    assert CENSOR_PRESETS["body"] == frozenset(
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
    )
    assert not any(c.startswith("FACE_") for c in CENSOR_PRESETS["body"])
    assert not any(c.endswith("_COVERED") for c in CENSOR_PRESETS["body"])


def test_body_is_superset_of_all():
    assert CENSOR_PRESETS["all"] <= CENSOR_PRESETS["body"]


def test_body_with_covered():
    assert resolve_censor_classes("body", covered=True) == CENSOR_PRESETS["body"] | frozenset(
        {
            "ANUS_COVERED",
            "BUTTOCKS_COVERED",
            "FEMALE_BREAST_COVERED",
            "FEMALE_GENITALIA_COVERED",
        }
    )


def test_resolve_face_male():
    result = resolve_censor_classes("female", face_male=True)
    assert "FACE_MALE" in result
    assert "FACE_FEMALE" not in result


def test_resolve_face_female():
    result = resolve_censor_classes("male", face_female=True)
    assert "FACE_FEMALE" in result
    assert "FACE_MALE" not in result


def test_resolve_feet():
    result = resolve_censor_classes("all", feet=True)
    assert "FEET_EXPOSED" in result


def test_resolve_buttocks():
    result = resolve_censor_classes("all", buttocks=True)
    assert "BUTTOCKS_EXPOSED" in result
    for cls in CENSOR_PRESETS["all"]:
        assert cls in result
    assert "BUTTOCKS_EXPOSED" not in resolve_censor_classes("all")


def test_resolve_none_with_buttocks():
    result = resolve_censor_classes("none", buttocks=True)
    assert result == frozenset({"BUTTOCKS_EXPOSED"})


def test_resolve_all_flags_combined():
    result = resolve_censor_classes(
        "all", covered=True, face_male=True, face_female=True, feet=True
    )
    assert "ANUS_COVERED" in result
    assert "BUTTOCKS_COVERED" in result
    assert "FEMALE_BREAST_COVERED" in result
    assert "FEMALE_GENITALIA_COVERED" in result
    assert "FACE_MALE" in result
    assert "FACE_FEMALE" in result
    assert "FEET_EXPOSED" in result
    # Original preset classes still present
    for cls in CENSOR_PRESETS["all"]:
        assert cls in result


def test_resolve_invalid_preset_raises():
    with pytest.raises(ValueError, match="unknown censor preset 'nonexistent'"):
        resolve_censor_classes("nonexistent")


# --- CensorConfig tests ---


def test_censor_config_resolve_matches_function():
    config = CensorConfig(preset="all", covered=True, feet=True)
    expected = resolve_censor_classes("all", covered=True, feet=True)
    assert config.resolve_classes() == expected


def test_censor_config_defaults():
    config = CensorConfig()
    assert config.resolve_classes() == CENSOR_PRESETS["all"]


def test_censor_config_buttocks():
    config = CensorConfig(preset="all", buttocks=True)
    assert config.resolve_classes() == resolve_censor_classes("all", buttocks=True)


# --- EraX class mapping contract test ---


def test_erax_mapped_classes_covered_by_presets():
    """All canonical class names emitted by EraX must be recognized by censor presets."""
    from bsafe.detector import _ERAX_CLASS_MAP

    all_preset_classes = set()
    for preset_classes in CENSOR_PRESETS.values():
        all_preset_classes.update(preset_classes)

    for erax_name, canonical_names in _ERAX_CLASS_MAP.items():
        for canonical in canonical_names:
            assert canonical in all_preset_classes, (
                f"EraX class '{erax_name}' maps to '{canonical}' which is not in any CENSOR_PRESETS"
            )
