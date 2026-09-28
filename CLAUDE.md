# Bsafe — Project Conventions

- Use `uv` for all package management (install, sync, run).
- Python >= 3.14. Format and lint with `ruff`.
- Run tests: `uv run pytest`
- Lint: `uv run ruff check bsafe/ tests/`
- Format: `uv run ruff format bsafe/ tests/`
- CLI entry point: `bsafe/cli.py` → `main()`
- Swift helper lives in `swift/`. Build with `cd swift && swift build -c release`.
- IPC protocol defined in `bsafe/protocol.py`. Python is the socket server, Swift connects as client.
- Keep `README.md` updated when adding user-facing features, CLI options, or setup steps.
- Keep `CLAUDE.md` updated when adding conventions, entry points, or architectural decisions.
- When adding or changing CLI flags/defaults, update the config template in `bsafe/config.py` (`generate_default_config()`) to match.

## Detection backends

`bsafe/detector.py` uses a backend abstraction: `_NudeNetBackend` (NudeNet ONNX models) and `_EraXBackend` (ultralytics YOLO `.pt` models). `Detector` delegates to the correct backend based on `ModelInfo.backend`. `ultralytics` is an optional dependency — install with `uv sync --extra erax`. EraX class names are mapped to NudeNet canonical names so `censor.py` works unchanged.

## Live pipeline

Live `bsafe start` uses frame credits (at most one outstanding per display):
Python sends `CMD_REQUEST_FRAME`, Swift replies with the newest raw BGRA frame
(`MSG_FRAME_RAW` + `MSG_DISPLAY_INFO`), Python runs `FullFrameNudeDetector`
(ONNX Runtime, CoreML EP, static shapes via free-dimension overrides) at native
resolution and replies with `CMD_CENSOR_SEQ` boxes tagged by frame `seq`.
Swift's native `BsafeCore` tracker moves boxes every captured frame, extends
them along motion, and owns persistence/smoothing; `BoxTracker`
(`bsafe/tracking.py`) is video-only. Matched detections blend per edge: an edge
moved outward snaps, an edge moved inward shrinks by `TrackerConfig.shrinkAlpha`
(0.1) per 0.1 s of detection-frame time (`1-(1-a)^(dt/0.1)`; per pass when the
same frame is re-applied). A track is removed only after `persistPasses` missed
passes **and** `TrackerConfig.minPersistSeconds` (1.0) since its last match, so
persistence does not shrink as detection gets faster. Both defaults (was 0.25 / 0.6) were chosen with `bsafe bench sweep` on scrolling-feed recordings: lower leak across all classes for a small over-censor cost. `--smooth-alpha` is still sent in `CMD_START` but only affects video.
Each track also keeps `core` (the latest matched detection, unblended); motion is
estimated on `estimationRegion(core)`, the central 55% (matching the suggested
0.4 per-side padding; the CLI default is 0.0), both per frame and when catching a detection up from its
`seq`, so a static margin cannot out-vote an object moving over a static
background. A core that crosses a frame edge falls back to the full core (the
central crop breaks edge coasting). Known gaps: near frame edges and with
`--full-censor` (3x boxes) the full or cropped region can still be mostly static.

`--min-padding` (live `start` and bench only, default 0) floors the per-side
padding in `censor.pad_box` to `max(int(w*padding), min_padding)` so tiny
detections still get a usable box; `--full-censor` expansion happens after.
Units are capture pixels, which on Retina are points (live captures 1728x1117 on a
3456x2234 panel), not device pixels.

`--extra-scales` (NudeNet full-frame path only, default `0.5`) adds extra
detection passes on the frame downscaled to each factor via
`fastdetect.detect_multiscale` (live and `bench detect` share the code;
factors are `s / detect_scale` for live point-size scales, frame-relative for
bench).

Without `-v`, the helper's stderr is a pipe that `StderrTail`
(`bsafe/swift_helper.py`) drains on a daemon thread, keeping the last 50
lines for the exit message. Never leave a helper pipe undrained: once it
fills (~64 KB), the helper blocks and the overlay freezes.
Capture interruptions auto-recover in-process per display with exponential backoff (`kill -USR1 <BsafeCapture pid>` simulates an interruption for testing); persistent failure still exits with code 3.

## Benchmark

Files: `bsafe/bench.py`, `swift/Sources/BsafeReplay`, `BsafeCore/ReplayScheduler.swift`.
`bsafe bench detect` writes per-frame detections (`dets.jsonl`); `replay` turns
them into censor boxes (same class filter + padding + merge as live) and hands
them to the `bsafe-replay` binary, which replays them through the live tracker
on the live frame-credit schedule, then scores against ground truth. Ground
truth = per-frame oracle detections, linked into objects and gap-filled.
Build with `cd swift && swift build -c release` (produces `bsafe-replay`).
Bench recordings must be scaled to the live capture size (even height) or
detection resolution/latency will not match live.
`bench replay --min-padding N` floors box padding and is recorded in
`score.json` (run dir gets a `-mpxN` suffix when nonzero); `bench sweep
--min-padding N` applies it to every sweep config. Bench writes only metadata
(never pixels). Scoring: GT matches the on-screen line (latest `display_pts`
<= pts, including `event: apply` lines rendered at completion time via
`ReplayScheduler.beginFrameTimed`); over-censor averages over every displayed
frame vs the dilated GT of the nearest dets frame (per-class null);
never-covered objects add span + 1 frame to `first_cover_ms` (+ p90); detect
pts use `CAP_PROP_POS_MSEC` with index/fps fallback, header fps
`(n-1)/last_pts`. `bench replay/sweep --gt-dets PATH` (repeatable, `--gt-conf`)
scores recall against the union of stronger GT dets (scaled to live size,
same-frame IoU>=0.5 merged, frame-count tolerance <=2); new metrics
`objects_missed_pct` / `objects_missed_by_live_pct` (live IoU>=0.1).

## Batch processing

`bsafe video` and `bsafe image` accept multiple files (shell globs). Process tree:

- **Video batch**: each file runs in a spawned subprocess (which itself spawns chunk subprocesses). Full memory isolation per file. `gc.collect()` between files.
- **Image batch**: all files run in the main process sequentially. `gc.collect()` between files (`process_image` creates and closes the detector internally). No subprocess overhead since images are lightweight.

Single-file mode preserves existing behavior exactly (no batch wrappers). `-o/--output` is disallowed with multiple inputs.

## Before committing

Always run all CI checks locally **before** every commit — format, lint, and tests. Fix any issues before committing. Do not prompt the user to commit; wait for them to ask.

```sh
uv run ruff format bsafe/ tests/
uv run ruff check bsafe/ tests/
uv run pytest
```

## Python 3.14 syntax

This project targets Python >= 3.14. Some syntax that was invalid or had different semantics in older Python versions is now standard:

- `except A, B:` is valid and means `except (A, B):` (PEP 758). Do **not** "fix" this by adding parentheses — ruff enforces the unparenthesized form. This is **not** the Python 2 `except A as B` pattern; that was removed in Python 3.0.

Trust ruff's formatting output for 3.14-era syntax questions.

## CLI output style

`bsafe/style.py` is the single source for all terminal styling. Semantic helpers:

- `dim()` — config lines, timestamps, secondary info
- `bold()` — status transitions, output paths, model names
- `error()` — red, only for "Error:" prefixes or failure markers
- `warn()` — yellow, only for "Warning:" or "WARN"/"NOT FOUND" tokens
- `success()` — green, "OK", "Saved", completion counts
- `info()` — cyan, chunk labels, file counts, timing durations
- `detection()` — magenta, `[detection]` log prefix in verbose mode
- `timestamp()` — dim `[HH:MM:SS]` string

Rules:
- Never color full lines — only short tokens get colored.
- Keep file paths uncolored (use `bold()` for emphasis on input/output paths, not color).
- Respect `NO_COLOR` env var and non-TTY output (auto-detected by `_color_enabled()`).
- No external dependencies — pure ANSI escape codes only.

## Menu bar app

`swift/Sources/BsafeMenuKit` is a UI-free library (settings-to-CLI-arguments mapping, stats-line parsing, supervised `bsafe start` child via posix_spawn with its own process group and SIGINT→SIGTERM→SIGKILL escalation, stdout/stderr drained on background threads); `swift/Sources/BsafeMenuBar` is the SwiftUI `MenuBarExtra` app that only supervises the CLI, with tests in `swift/Tests/BsafeMenuKitTests`. Install with `scripts/install-menubar.sh`, which builds release, assembles `~/Applications/Bsafe.app` (Info.plist carries `BsafeCLIPath`), and ad-hoc signs it.

## Privacy

Screen data is personal data. Strict rules:

- **No network calls.** All processing (capture, inference, overlay) must happen locally. Never add HTTP clients, telemetry, analytics, or any outbound connections.
- **No persistence of screen content.** Frames must stay in memory (or short-lived temp files required by libraries) and never be saved to disk intentionally. Temp files must be cleaned up.
- **No logging of screen content.** Never log raw pixel data, file paths of saved frames, or detection image crops. Logging detection metadata (class, confidence, box coordinates) is acceptable.
