# Bsafe

## About

Censor NSFW content on your macOS screen in real time. CLI tool, fully offline.

Supports real-time screen censoring, video file processing (with audio
preservation), and image censoring — with multiple detection models,
censor styles (black boxes, blur, pixelation, text overlay), and
per-body-part configuration.

## Setting up locally

Requires macOS 14+, Python 3.14, and [uv](https://docs.astral.sh/uv/):

```sh
uv run bsafe bootstrap
```

This installs Python dependencies (`uv sync`) and builds the Swift
screen-capture helper. At the end it prints an alias line you can add
to your `~/.zshrc` to make `bsafe` globally available.

You must grant **Screen Recording** permission to your terminal app
(System Settings → Privacy & Security → Screen Recording).

## Usage

Check that everything is in order:

```sh
bsafe doctor
```

Start real-time screen censoring (no daemonized support yet):

```sh
bsafe start
```

### Display selection

By default `bsafe start` captures only the **primary** display. Most setups
have one primary and one secondary display — the aliases `primary` and
`secondary` are enough to pick either without looking up IDs.

List connected displays:

```sh
bsafe displays
```

Target a specific display:

```sh
bsafe start --display secondary   # first non-primary display
bsafe start --display 1234567     # by numeric ID (from bsafe displays)
bsafe start --display all         # all displays (uses more CPU)
```

Covering more displays at higher resolutions with more NSFW content increases CPU load
and may reduce quality — you are processing and rendering video in real time.

## Menu bar app

A tiny menu bar app that starts/stops `bsafe start` and exposes a few
settings. It only supervises the CLI — capture and detection stay in
`bsafe start`.

Install:

```sh
scripts/install-menubar.sh
```

This builds the Swift targets, assembles `~/Applications/Bsafe.app`
(ad-hoc signed), and opens it. On first Start, grant **Screen Recording**
permission to Bsafe (System Settings → Privacy & Security → Screen
Recording). The script signs with your Developer ID / Apple Development
identity when one exists (override with `BSAFE_SIGN_IDENTITY`), so grants
survive reinstalls; without one it signs ad-hoc and grants reset on every
reinstall.

Settings map to `bsafe start` flags: Censor → `--censor`, Feet →
`--feet`/`--no-feet`, Style → `--pixels`/`--blur`, Padding → `--padding`,
Min padding → `--min-padding`, Quality → Fast (`--extra-scales none`),
Standard (`--extra-scales 0.5`), Strict (`--detect-scale 1.5
--extra-scales 0.5`).

If `bsafe start` exits unexpectedly while switched on, the app relaunches it
(2/5/10/30 s backoff) and gives up after 5 failures within 5 minutes.

Uninstall: quit Bsafe, then `rm -rf ~/Applications/Bsafe.app`.

### Process video files

Produce censored copies of local videos (originals are never modified):

```sh
bsafe video clip.mp4                  # → clip.320n.bsafe.mp4
bsafe video clip.mp4 --blur           # blur instead of black boxes
bsafe video clip.mp4 --pixels         # pixelation effect
bsafe video clip.mp4 --censor-text    # overlay "NSFW" text
bsafe video clip.mp4 --fps 5          # run detection at 5 FPS (output keeps native FPS)
bsafe video clip.mp4 --enhance dim    # low-light enhancement (denoise + adaptive gamma)
```

Process multiple files at once (shell globs work):

```sh
bsafe video *.mp4 --blur              # process all .mp4 files with blur
bsafe video a.mp4 b.mov c.m4v        # explicit file list
```

Supported formats: `.mp4`, `.m4v`, `.mov`. If `ffmpeg` is installed, audio is
preserved in the output; otherwise the video is written without audio.

Videos are processed in chunks (default: 5000 frames). If interrupted, re-run the
same command to resume — completed chunks are skipped automatically. In batch mode,
files with existing output are skipped (idempotent). Non-video files and directories
are filtered out automatically.

### Process image files

Produce censored copies of local images:

```sh
bsafe image photo.jpg                 # → photo.320n.bsafe.jpg
bsafe image *.jpg --pixels            # process all .jpg files with pixelation
bsafe image a.png b.webp --blur       # explicit file list
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.
Re-running the same command skips files that already have output (idempotent).

Note: `-o/--output` cannot be used with multiple input files.

### Configuration file

`bsafe bootstrap` creates `~/.config/bsafe/config.toml` with all default values.
Edit this file to set your preferred defaults — CLI flags always override it.

Sections: `[start]` for start-only flags, `[video]` for video-only flags,
`[common]` for shared flags used by both commands.

### Options

Shared flags (work with both `start` and `video`):

- `--confidence N` — minimum detection confidence, 0.0–1.0 (default: 0.0 for NudeNet, 0.2 for EraX)
- `--censor {none,female,male,all,body}` — what to censor (default: all). `body` censors every exposed body part except faces (adds buttocks, feet, belly, armpits and male chest to `all`). Use `none` to skip nudity censoring while still using additive flags like `--face-male`
- `--padding N` — box expansion fraction (default: 0.0)
- `--persist-frames N` — frames a box persists after disappearing (default: 8). In live `start` these are detection passes, and a box also stays at least 1.0 s after its last detection
- `--smooth-alpha N` — EMA smoothing weight for `video` box tracking, 0.0–1.0 (default: 0.5). Live `start` ignores it (see "How live mode works")
- `--blur [N]` — blur effect (default intensity: 1.0)
- `--pixels [N]` — pixelation effect (default intensity: 1.0)
- `--censor-text [TEXT]` — overlay text on censored regions (default: "NSFW")
- `--model {320n,640m,erax-nano,erax-small,erax-medium}` — detection model (default: `320n`). See model details below
- `--covered` — also censor covered body parts (anus, buttocks; breasts when `--censor` is `female` or `all`)
- `--face-male` — also censor male faces
- `--face-female` — also censor female faces
- `--feet` — also censor exposed feet
- `--buttocks` — also censor exposed buttocks (off by default: more false positives, e.g. tight clothing)
- `--full-censor` — expand censor area by 3x
- `-v` / `--verbose` — enable debug logging

`start`-only flags:

- `--fps N` — capture FPS; up to the display refresh rate, e.g. 120 on ProMotion (default: 60)
- `--display VALUE` — display to capture: `primary`, `secondary`, `all`, or numeric ID (default: `primary`)
- `--dry-run` — run the loop without the Swift helper or detector
- `--stats` — log live detection timing every ~2s plus native capture/tracking stats
- `--detect-scale N` — detection resolution as a multiple of the display's point size (default 1.0). 1.5 detects images down to ~100 pt wide at ~2x GPU cost
- `--min-padding N` — minimum padding in capture pixels per side, added to small boxes (default: 0). Units are the capture size printed at startup (`Display 1: capturing 1728x1117 px`). Small detections carry most of the leaked area and proportional `--padding` barely grows them: on scrolling X feed recordings at live capture size, a 24 px floor cut leak by up to 20% for about +0.15% of screen over-censor
- `--extra-scales LIST` — extra detection passes at these multiples of point size, e.g. `0.5` or `0.5,0.75` (each must be < `--detect-scale`; default `0.5` for NudeNet, `none` to disable). The 320n model was trained at 320 px input and misses large close-ups full-frame at native 1728 px: one extra 0.5x pass cut leak vs a 640m oracle from 18.9% to 10.3% (feet), 50.7% to 32.5% (armpits), 17.5% to 16.5% (butt) for ~+5 ms per frame. Strict mode: `bsafe start --detect-scale 1.5` keeps the extra 0.5 pass and additionally halves small-object misses at ~2x total cost. NudeNet-only (warned and ignored with EraX)

### How live mode works

Each display is captured natively and every captured frame feeds a native
per-display tracker (`BsafeCore`, pure Swift) that moves existing censor boxes
with the content and re-renders the overlay at capture rate, extended along the
motion direction to cover display latency.

Python grants one frame credit per display. Swift answers with the newest raw
BGRA frame. Python runs full-frame NudeNet at the frame's native resolution on
the GPU (ONNX Runtime CoreML EP, static shapes), builds censor boxes, and
replies tagged with the frame `seq`. Swift catches the boxes up from `seq` to
its newest frame and merges them into its tracks.

`--detect-scale` trades resolution for GPU cost: it scales the capture as a
multiple of the display's point size before detection. `1.0` is cheapest;
`1.5` detects images down to ~100 pt wide at ~2x GPU cost (EraX models require
`1.0`). The native tracker owns live tracking and persistence
(`--persist-frames` tunes how many missed detection passes a track survives;
a track is also kept for at least 1.0 s after its last match, so a fast detector
cannot drop a box on a short miss). Python no longer smooths live boxes. When a
new detection matches a track, each box edge that the detection moves outward
snaps to it at once, and each edge it moves inward shrinks by a tenth of the
difference per 0.1 s of detection time, independent of the detection rate. A detection that briefly comes back
narrower (common while an image scrolls into view) therefore cannot uncover
part of the region. `--smooth-alpha` does not apply to live mode.

Suggested setup for a 120 Hz Mac:

```sh
bsafe start --fps 120 --detect-scale 1.5 --padding 0.4 --persist-frames 4
```

Limitation: an overlay reacts after content is drawn, so the first frames of
newly appearing content can still be visible (typically a few tens of
milliseconds), and detection quality depends on the model.

Troubleshooting: capture interruptions (screen lock/sleep, permission re-confirm) auto-recover per display; if recovery fails persistently the helper still exits (code 3).

`image`-only flags:

- `-o` / `--output PATH` — output file path (default: `<input>.<model>.bsafe.<ext>`)

`video`-only flags:

- `-o` / `--output PATH` — output file path (default: `<input>.<model>.bsafe.<ext>`)
- `--fps N` — detection FPS override (default: native video FPS)
- `--chunk-frames N` — frames per processing chunk (default: 5000). Smaller chunks use less memory but may cause brief tracking gaps at chunk boundaries.
- `--enhance dim` — low-light enhancement for dim-but-visible footage. Pre-scans the video, applies FFmpeg temporal denoising (`hqdn3d`), then adaptive per-frame gamma/contrast correction. Automatically skips enhancement for already-bright footage. Requires `ffmpeg` for denoising (enhancement still works without it).

### Benchmark (dev)

Frozen baselines live in `bench/baselines/` (metrics and config only; recordings stay
local). `x-feed-2026-09-27.json` is the reference for the current live pipeline; compare
new runs against it before changing detection or tracking defaults.

`bsafe bench` replays a screen recording through the live tracker offline to
measure leak vs. over-censor for different padding/tracker settings:

```sh
bsafe bench detect rec.mov -o bench/rec   # per-frame detections → dets.jsonl
bsafe bench replay bench/rec              # boxes → Swift replay → score.json
bsafe bench sweep bench/rec               # grid of configs → sweep.json table
```

Recordings are user-made and stay local. Record with Bsafe stopped (so no overlay is
captured), then scale to the live capture size printed by `bsafe start`
(e.g. 1728x1117 on a 3456x2234 Retina panel) with an even height, otherwise
detection runs at a different resolution and latency than live:

```sh
ffmpeg -f avfoundation -capture_cursor 0 -framerate 60 -i "<screen>:none" -t 48 -r 60 -fps_mode cfr raw.mov
ffmpeg -i raw.mov -vf scale=1728:1116:flags=area rec.mov
```

Use `bench replay --overhead-ms` to match live `avg_receive_to_send_ms` minus the
offline detect time if they differ.
The bench writes only detection metadata and box coordinates — never pixels,
crops, or frames.

Scoring: GT frames match the displayed line on screen at their pts (latest
`display_pts` <= pts, including `event: apply` re-renders emitted at detection
completion time). Over-censor averages over every displayed frame against the
dilated GT of the nearest dets frame (empty GT counts fully as excess);
per-class `over_censor_pct` is null (displayed boxes carry no class).
Never-covered objects contribute their full span + 1 frame to `first_cover_ms`
(median + `first_cover_p90_ms`). Frame pts come from `CAP_PROP_POS_MSEC`
(first frame 0, fallback to index/fps); header fps is `(n-1)/last_pts`.

To measure recall of the live detector, score against a stronger GT source
with `--gt-dets` (repeatable; `--gt-conf` applies to all GT sources, default
0.25). When given, GT is the union of the listed `dets.jsonl` files — the
live `DIR/dets.jsonl` is NOT included unless listed — scaled to the live
resolution and linked/gap-filled as usual. For example, run `bench detect
--model 640m` on the full-resolution recording (plus `320n` for a second
opinion) and replay the live-scale dir against both:

```sh
bsafe bench replay bench/rec --gt-dets bench/hires-640m/dets.jsonl --gt-dets bench/hires-320n/dets.jsonl
bsafe bench sweep bench/rec --gt-dets bench/hires-640m/dets.jsonl
```

New recall metrics (overall and per class): `objects_missed_pct` (% of GT
objects never covered) is the headline recall number, and
`objects_missed_by_live_pct` (% of GT objects the live detections never saw
with same-class IoU >= 0.1) separates "model never saw it" from
"tracker/latency failed". `score.json` records `gt_sources` and `gt_conf`.

To benchmark `--extra-scales`, run `bench detect --extra-scales 0.5` on a
recording already scaled to the live capture size (factors here are
frame-relative, and the header records them). To benchmark strict mode
(`--detect-scale 1.5` with extra `0.5`), record/scale the recording to 1.5x
the live size and use factor `0.333` (= 0.5 / 1.5).

### Detection models

The default `320n` model is bundled with NudeNet. Other models require a manual download.

| Model | Backend | Size | Notes |
|-------|---------|------|-------|
| `320n` | NudeNet | bundled | Fast, default |
| `640m` | NudeNet | ~90 MB | More accurate |
| `erax-nano` | EraX YOLO | ~5 MB | Fastest EraX, mAP 0.438 |
| `erax-small` | EraX YOLO | ~40 MB | Balanced, mAP 0.453 |
| `erax-medium` | EraX YOLO | ~19 MB | Best EraX accuracy, mAP 0.467 |

NudeNet models have broader coverage (faces, covered parts, feet, buttocks) and are lightweight —
best for real-time `bsafe start`. EraX models are more targeted (e.g. nipple-specific)
and heavier — better suited for `bsafe image` and `bsafe video` where FPS isn't a constraint.

#### Using the 640m model

```sh
mkdir -p ~/.config/bsafe/models
curl -Lo ~/.config/bsafe/models/640m.onnx \
  https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx
bsafe start --model 640m
```

#### Using EraX models

EraX models use [ultralytics](https://github.com/ultralytics/ultralytics) YOLO and require an extra dependency:

```sh
uv sync --extra erax
```

Download a model (e.g. `erax-nano`):

```sh
mkdir -p ~/.config/bsafe/models
curl -Lo ~/.config/bsafe/models/erax-anti-nsfw-yolo11n-v1.1.pt \
  https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11n-v1.1.pt
```

Other variants:

```sh
# erax-small
curl -Lo ~/.config/bsafe/models/erax-anti-nsfw-yolo11s-v1.1.pt \
  https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11s-v1.1.pt

# erax-medium
curl -Lo ~/.config/bsafe/models/erax-anti-nsfw-yolo11m-v1.1.pt \
  https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11m-v1.1.pt
```

Then use with `--model`:

```sh
bsafe start --model erax-nano
bsafe image photo.jpg --model erax-small
bsafe video clip.mp4 --model erax-medium
```

## Privacy & censor strength

**Real-time screen mode** (`bsafe start`) draws a transparent overlay window on
top of your screen. It does not modify the underlying applications or websites
in any way — censoring is purely visual and disappears when bsafe stops.

**Video and image modes** (`bsafe video`, `bsafe image`) produce new files with
the censored pixels baked into the output. The original file is never modified.
However, not all censor styles destroy information equally:

- **Black boxes** (default) replace every pixel in the censored region with
  solid black. The original data is completely destroyed — recovery is
  impossible regardless of technique or computing power.
- **Blur** (`--blur`) applies a Gaussian blur that removes high-frequency
  detail. At default or higher intensity this is practically irreversible, but
  the exact kernel parameters are deterministic from the box dimensions and
  intensity (both visible or inferable from the output). At low intensity,
  deconvolution techniques can partially recover edges and shapes.
- **Pixelation** (`--pixels`) downscales each region and scales it back up,
  producing uniform color blocks. Each block preserves the average color of the
  original pixels, retaining more information than the other modes. Published
  machine-learning attacks have demonstrated recovering recognizable faces and
  text from pixelated images. This is the weakest censor mode.

If your priority is ensuring censored content cannot be recovered from the
output file, use black boxes (the default). If you use blur or pixelation for
aesthetic reasons, consider using higher intensity values (e.g. `--blur 3`,
`--pixels 3`) to reduce the amount of recoverable information.

When using `--fps` to skip detection frames in video mode, content that first
appears between detection frames will go uncensored for a few frames until the
next detection cycle picks it up. The default `--persist-frames` setting keeps
boxes active across gaps, but cannot predict content that hasn't been seen yet.

## License

Under [MIT License](./LICENSE).

Copyright (c) 2026 Marcell "Mazuh" G. C. da Silva.
