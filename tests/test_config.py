"""Config template tests (no I/O)."""

from bsafe.cli import _build_parser
from bsafe.config import (
    VALID_KEYS,
    generate_default_config,
    load_config,
    normalize_extra_scales,
)


def test_start_keys_in_template():
    template = generate_default_config()
    for key in VALID_KEYS["start"]:
        assert key in template, f"key {key!r} missing from default config template"


def test_start_valid_keys():
    assert (
        VALID_KEYS["start"]
        == {
            "fps",
            "dry_run",
            "display",
            "stats",
            "detect_scale",
            "min_padding",
            "extra_scales",
        }
        | VALID_KEYS["common"]
    )
    for removed in (
        "max_frame_age_ms",
        "inference_resolution",
        "detail_scan",
        "motion_compensation",
        "motion_lookahead_ms",
    ):
        assert removed not in VALID_KEYS["start"]
        assert removed not in generate_default_config()


def test_min_padding_is_start_only():
    assert "min_padding" in VALID_KEYS["start"]
    assert "min_padding" not in VALID_KEYS["common"]
    assert "min_padding" not in VALID_KEYS["video"]
    assert "min_padding" not in VALID_KEYS["image"]
    assert "min_padding" in generate_default_config()


def test_min_padding_config_applies_as_start_default(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[start]\nmin_padding = 48\n", encoding="utf-8")
    config = load_config(str(cfg))
    assert config["start"]["min_padding"] == 48
    parser, start_parser, _, _ = _build_parser()
    start_config = {**config.get("common", {}), **config.get("start", {})}
    start_parser.set_defaults(**start_config)
    args = parser.parse_args(["start"])
    assert args.min_padding == 48


def test_extra_scales_is_start_only():
    assert "extra_scales" in VALID_KEYS["start"]
    assert "extra_scales" not in VALID_KEYS["common"]
    assert "extra_scales" not in VALID_KEYS["video"]
    assert "extra_scales" not in VALID_KEYS["image"]
    assert "extra_scales" in generate_default_config()


def test_normalize_extra_scales_forms():
    import pytest

    assert normalize_extra_scales(None) is None
    assert normalize_extra_scales("0.5") == "0.5"
    assert normalize_extra_scales("none") == "none"
    assert normalize_extra_scales(0.5) == "0.5"
    assert normalize_extra_scales([0.5, 0.75]) == "0.5,0.75"
    with pytest.raises(ValueError):
        normalize_extra_scales(True)
    with pytest.raises(ValueError):
        normalize_extra_scales(False)
    with pytest.raises(ValueError):
        normalize_extra_scales(["none"])
    with pytest.raises(ValueError):
        normalize_extra_scales([0.5, "none"])
    with pytest.raises(ValueError):
        normalize_extra_scales([True])
    with pytest.raises(ValueError):
        normalize_extra_scales({"s": 0.5})


def test_load_config_invalid_extra_scales_warns_and_drops(tmp_path, capsys):
    for body in (
        "extra_scales = true",
        'extra_scales = ["none"]',
        'extra_scales = [0.5, "none"]',
    ):
        cfg = tmp_path / "config.toml"
        cfg.write_text(f"[start]\n{body}\n", encoding="utf-8")
        config = load_config(str(cfg))
        assert "extra_scales" not in config.get("start", {})
        err = capsys.readouterr().err
        assert "Warning" in err
        assert "extra_scales" in err
    # Dropped key means the CLI default applies and other commands keep working.
    cfg = tmp_path / "config.toml"
    cfg.write_text("[start]\nextra_scales = true\n", encoding="utf-8")
    config = load_config(str(cfg))
    capsys.readouterr()
    parser, start_parser, _, _ = _build_parser()
    start_config = {**config.get("common", {}), **config.get("start", {})}
    start_parser.set_defaults(**start_config)
    args = parser.parse_args(["start"])
    assert args.extra_scales is None


def test_extra_scales_config_float_list_string(tmp_path):
    from bsafe.live import parse_extra_scales

    for body, expected in [
        ('extra_scales = "0.5"', (0.5,)),
        ("extra_scales = 0.5", (0.5,)),
        ("extra_scales = [0.5, 0.75]", (0.5, 0.75)),
        ('extra_scales = "none"', ()),
    ]:
        cfg = tmp_path / "config.toml"
        cfg.write_text(f"[start]\n{body}\n", encoding="utf-8")
        config = load_config(str(cfg))
        raw = config["start"]["extra_scales"]
        assert isinstance(raw, str)
        assert parse_extra_scales(raw) == expected
        parser, start_parser, _, _ = _build_parser()
        start_config = {**config.get("common", {}), **config.get("start", {})}
        start_parser.set_defaults(**start_config)
        args = parser.parse_args(["start"])
        assert parse_extra_scales(args.extra_scales) == expected
