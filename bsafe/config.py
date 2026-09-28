import os
import sys
import tomllib

from bsafe.style import warn

CONFIG_PATH = os.path.expanduser("~/.config/bsafe/config.toml")

_COMMON_KEYS = {
    "confidence",
    "censor",
    "padding",
    "persist_frames",
    "smooth_alpha",
    "blur",
    "pixels",
    "censor_text",
    "full_censor",
    "covered",
    "face_male",
    "face_female",
    "feet",
    "buttocks",
    "model",
    "verbose",
}

VALID_KEYS = {
    "start": {"fps", "dry_run", "display", "stats", "detect_scale", "min_padding", "extra_scales"}
    | _COMMON_KEYS,
    "video": {"fps", "chunk_frames", "enhance"} | _COMMON_KEYS,
    "image": _COMMON_KEYS.copy(),
    "common": _COMMON_KEYS,
}


def normalize_extra_scales(value) -> str | None:
    """Normalize a TOML extra_scales value to the string form the CLI parses.

    Raises ValueError on invalid values (bool, non-numeric list items, or
    unsupported types); callers warn and drop the key so the default applies.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"invalid extra_scales {value!r}: expected string or numbers")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    if isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"invalid extra_scales {value!r}: list items must be numbers")
        return ",".join(f"{float(v):g}" for v in value)
    raise ValueError(f"invalid extra_scales {value!r}: expected string or numbers")


def generate_default_config() -> str:
    return """\
# Bsafe configuration — edit values to change defaults.
# CLI flags always override these values.
# Remove a key (or set to its zero value) to use the hardcoded default.

[start]
fps = 60
display = "primary"
dry_run = false
# min_padding = 0     # capture pixels (see startup log); try 24
# model =            # override [common] model for real-time screen censoring
# stats = false      # log live detection timing every ~2s plus native capture/tracking stats
# detect_scale = 1.0  # detection resolution multiple: 1.0, 1.25, 1.5, 2.0 (NudeNet only; 1.5 catches ~100 pt images)
# extra_scales = "0.5"   # comma list of extra detection scales; "none" disables

[video]
# fps =              # unset = use native video FPS
# chunk_frames =     # unset = 5000
# enhance =          # unset = disabled. Options: dim
# model =            # override [common] model for video processing

[image]
# model =            # override [common] model for image processing
# Note: persist_frames and smooth_alpha from [common] are ignored for images.

[common]
# confidence =       # unset = 0.0 for NudeNet, 0.2 for EraX
# censor = "all"       # Options: none, female, male, all, body
censor = "all"
padding = 0.0
persist_frames = 8
smooth_alpha = 0.5     # video only; live boxes grow at once and shrink gradually
blur = 0.0
pixels = 0.0
# censor_text =      # unset = disabled; set to "NSFW" or custom string
full_censor = false
covered = false
face_male = false
face_female = false
feet = false
buttocks = false
# model =            # unset = "320n". Options: 320n, 640m, erax-nano, erax-small, erax-medium
verbose = false
"""


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load config from TOML file. Returns {} if missing or malformed."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        print(f"{warn('Warning:')} could not parse {path}: {e}")
        return {}
    for section, keys in config.items():
        if section not in VALID_KEYS:
            print(f"{warn('Warning:')} unknown config section [{section}] in {path}")
            continue
        if isinstance(keys, dict):
            for key in keys:
                if key not in VALID_KEYS[section]:
                    print(f"{warn('Warning:')} unknown key '{key}' in [{section}] in {path}")
    if (
        "start" in config
        and isinstance(config["start"], dict)
        and "extra_scales" in config["start"]
    ):
        try:
            config["start"]["extra_scales"] = normalize_extra_scales(
                config["start"]["extra_scales"]
            )
        except (ValueError, TypeError) as e:
            print(
                f"{warn('Warning:')} invalid [start] extra_scales in {path}: {e}",
                file=sys.stderr,
            )
            del config["start"]["extra_scales"]
    return config
