import argparse
import faulthandler
import gc
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time

from bsafe.style import bold, dim, error, info, success, warn

logger = logging.getLogger(__name__)


def _add_censor_args(parser):
    """Add shared censor flags to a subcommand parser."""
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Min detection confidence (default: 0.0 for NudeNet, 0.2 for EraX)",
    )
    parser.add_argument(
        "--censor",
        choices=["none", "female", "male", "all", "body"],
        default="all",
        help="What to censor: none, female, male, all, or body (all exposed body parts except faces) (default: all)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Box expansion fraction (default: 0.0)",
    )
    parser.add_argument(
        "--blur",
        nargs="?",
        type=float,
        const=1.0,
        default=0.0,
        help="Use blur instead of black rectangle; 1.0 = 100%% blur, >1 multiplies effect (default: 1.0)",
    )
    parser.add_argument(
        "--pixels",
        nargs="?",
        type=float,
        const=1.0,
        default=0.0,
        help="Use pixelation instead of black rectangle; 1.0 = 100%% pixelation, >1 multiplies effect (default: 1.0)",
    )
    parser.add_argument(
        "--censor-text",
        nargs="?",
        const="NSFW",
        default=None,
        help='Show text on censored region (default: "NSFW" if flag present with no value)',
    )
    parser.add_argument(
        "--full-censor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expand censor area by 3x (9x area)",
    )
    parser.add_argument(
        "--covered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor covered body parts (anus, buttocks; breasts when female/all)",
    )
    parser.add_argument(
        "--face-male",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor male faces",
    )
    parser.add_argument(
        "--face-female",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor female faces",
    )
    parser.add_argument(
        "--feet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor exposed feet",
    )
    parser.add_argument(
        "--buttocks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor exposed buttocks (off by default: more false positives, e.g. tight clothing)",
    )
    parser.add_argument(
        "--model",
        choices=["320n", "640m", "erax-nano", "erax-small", "erax-medium"],
        default=None,
        help="Detection model: '320n' (default, bundled), '640m' (accurate, requires download), "
        "'erax-nano/small/medium' (EraX YOLO, requires download + uv sync --extra erax)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def _add_temporal_args(parser):
    """Add temporal tracking flags (only for live/video, not image)."""
    parser.add_argument(
        "--persist-frames",
        type=int,
        default=8,
        help="Frames a box persists after disappearing (default: 8)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="EMA weight for video box smoothing, 0.0-1.0 (default: 0.5); live start ignores it",
    )


def _validate_censor_args(args):
    """Validate shared censor arguments. Exits on error."""
    if args.padding < 0:
        print(f"{error('Error:')} --padding must be >= 0", file=sys.stderr)
        sys.exit(1)
    if getattr(args, "min_padding", 0) < 0:
        print(f"{error('Error:')} --min-padding must be >= 0", file=sys.stderr)
        sys.exit(1)
    if hasattr(args, "smooth_alpha") and not (0.0 <= args.smooth_alpha <= 1.0):
        print(f"{error('Error:')} --smooth-alpha must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    blur = args.blur or 0.0
    pixels = args.pixels or 0.0
    if not (0.0 <= blur <= 3.0):
        print(f"{error('Error:')} --blur must be between 0.0 and 3.0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= pixels <= 3.0):
        print(f"{error('Error:')} --pixels must be between 0.0 and 3.0", file=sys.stderr)
        sys.exit(1)
    if blur > 0.0 and pixels > 0.0:
        print(f"{error('Error:')} --blur and --pixels are mutually exclusive", file=sys.stderr)
        sys.exit(1)


def _resolve_confidence(args):
    """Set confidence to a backend-appropriate default when not explicitly provided."""
    if args.confidence is not None:
        return
    from bsafe.detector import get_model_backend

    backend = get_model_backend(args.model)
    args.confidence = 0.2 if backend == "erax" else 0.0


def _warn_erax_unsupported(args):
    """Print warnings for flags that have no effect with EraX models."""
    from bsafe.detector import ERAX_UNSUPPORTED_FLAGS, get_model_backend

    if get_model_backend(args.model) != "erax":
        return
    for attr, flag in ERAX_UNSUPPORTED_FLAGS.items():
        if getattr(args, attr, False):
            print(
                f"{warn('Warning:')} {flag} has no effect with EraX models (no covered/face/feet/buttocks classes)",
                file=sys.stderr,
            )


def _extra_scales_display(args) -> str:
    """Format the resolved --extra-scales value for the Config line."""
    from bsafe.detector import get_model_backend
    from bsafe.live import parse_extra_scales

    raw = getattr(args, "extra_scales", None)
    try:
        parsed = parse_extra_scales(raw)
    except ValueError:
        return str(raw)
    if parsed is None:
        try:
            backend = get_model_backend(getattr(args, "model", None))
        except ValueError:
            return "0.5"
        return "none" if backend == "erax" else "0.5"
    if not parsed:
        return "none"
    try:
        backend = get_model_backend(getattr(args, "model", None))
    except ValueError:
        backend = "nudenet"
    if backend == "erax":
        return "none"
    return ",".join(f"{v:g}" for v in parsed)


def _resolve_extra_scales_for_start(args, detect_scale: float) -> tuple[float, ...]:
    """Resolve point-size extra scales and convert to frame-relative factors."""
    from bsafe.detector import get_model_backend
    from bsafe.live import parse_extra_scales, validate_extra_scales

    raw = getattr(args, "extra_scales", None)
    parsed = parse_extra_scales(raw)
    if parsed is None:
        backend = get_model_backend(getattr(args, "model", None))
        resolved = () if backend == "erax" else (0.5,)
    else:
        backend = get_model_backend(getattr(args, "model", None))
        if backend == "erax":
            resolved = ()
        else:
            validate_extra_scales(parsed, detect_scale)
            resolved = parsed
    return tuple(s / detect_scale for s in resolved)


def _validate_live_args(args):
    """Validate start-only live flags. Exits on error before the helper starts."""
    from bsafe.live import validate_live_options

    scale = getattr(args, "detect_scale", 1.0)
    if scale is None:
        scale = 1.0
        args.detect_scale = scale
    raw_extra = getattr(args, "extra_scales", None)
    try:
        validate_live_options(getattr(args, "model", None), scale, raw_extra)
    except ValueError as e:
        print(f"{error('Error:')} {e}", file=sys.stderr)
        sys.exit(1)


def _print_config(args):
    """Print active censor configuration as a sanity check."""
    if args.censor_text:
        censor_method = f"text ({args.censor_text})"
    elif args.blur:
        censor_method = f"blur ({args.blur})"
    elif args.pixels:
        censor_method = f"pixels ({args.pixels})"
    else:
        censor_method = "bar"
    parts = [
        f"model={args.model or '320n'}",
        f"confidence={args.confidence}",
        f"censor={args.censor}",
        f"method={censor_method}",
        f"padding={args.padding}",
    ]
    if hasattr(args, "min_padding"):
        parts.append(f"min_padding={args.min_padding}")
    extras = [
        name
        for name, enabled in [
            ("covered", args.covered),
            ("face_male", args.face_male),
            ("face_female", args.face_female),
            ("feet", args.feet),
            ("buttocks", args.buttocks),
            ("full_censor", args.full_censor),
        ]
        if enabled
    ]
    if extras:
        parts.append(f"extras={','.join(extras)}")
    if hasattr(args, "enhance") and args.enhance:
        parts.append(f"enhance={args.enhance}")
    if hasattr(args, "detect_scale"):
        parts.append(f"detect_scale={args.detect_scale}")
    if hasattr(args, "extra_scales"):
        parts.append(f"extra_scales={_extra_scales_display(args)}")
    if getattr(args, "stats", False):
        parts.append("stats=true")
    print(f"{dim('Config:')} {', '.join(parts)}", flush=True)


def cmd_start(args):
    if args.dry_run:
        stop = False

        def _handle_sigint(sig, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _handle_sigint)
        print(bold("Running...") + " press Ctrl+C to stop.", flush=True)
        while not stop:
            time.sleep(0.2)
        print(f"\n{bold('Stopped.')}")
        return

    from bsafe.censor import resolve_censor_classes
    from bsafe.detector import Detector, get_model_backend, resolve_model
    from bsafe.ipc import FrameServer
    from bsafe.live import LiveSession, LiveStats
    from bsafe.swift_helper import StderrTail, spawn_helper

    fps = args.fps
    if fps < 1 or fps > 255:
        print(f"{error('Error:')} --fps must be between 1 and 255", file=sys.stderr)
        sys.exit(1)
    _validate_censor_args(args)
    _validate_live_args(args)
    _resolve_confidence(args)
    socket_path = os.path.join(tempfile.gettempdir(), f"bsafe-{os.getpid()}.sock")

    censor_classes = resolve_censor_classes(
        args.censor,
        covered=args.covered,
        face_male=args.face_male,
        face_female=args.face_female,
        feet=args.feet,
        buttocks=args.buttocks,
    )
    padding = args.padding
    show_stats = bool(getattr(args, "stats", False))
    detect_scale_raw = getattr(args, "detect_scale", 1.0)
    if detect_scale_raw is None:
        detect_scale_raw = 1.0
    detect_scale = float(detect_scale_raw)
    stats = LiveStats() if show_stats else None

    # Warn about unsupported flags with EraX
    _warn_erax_unsupported(args)
    from bsafe.live import parse_extra_scales as _parse_extra

    try:
        _parsed_extra = _parse_extra(getattr(args, "extra_scales", None))
    except ValueError:
        _parsed_extra = ()
    try:
        _is_erax_model = get_model_backend(getattr(args, "model", None)) == "erax"
    except ValueError:
        _is_erax_model = False
    if _is_erax_model and _parsed_extra is not None and len(_parsed_extra) > 0:
        print(
            f"{warn('Warning:')} --extra-scales has no effect with EraX models (NudeNet-only)",
            file=sys.stderr,
        )

    _print_config(args)

    # Initialize detector: NudeNet uses the full-frame GPU path, EraX the classic one.
    print(f"Loading model {bold(args.model or '320n')}...", flush=True)
    backend = get_model_backend(args.model)
    is_nudenet = backend != "erax"
    if is_nudenet:
        from bsafe.fastdetect import FullFrameNudeDetector

        detector = FullFrameNudeDetector(
            model_path=resolve_model(args.model).path,
            min_confidence=args.confidence,
        )
    else:
        detector = Detector(min_confidence=args.confidence, model=args.model)

    # Resolve user-input-derived values before spawning the helper so nothing
    # that can raise on user input runs between spawn_helper and try/finally.
    extra_factors = _resolve_extra_scales_for_start(args, detect_scale)
    if not is_nudenet:
        extra_factors = ()
    min_padding_int = int(getattr(args, "min_padding", 0))

    # Start IPC server
    server = FrameServer(
        socket_path,
        fps=fps,
        blur=args.blur,
        pixels=args.pixels,
        censor_text=args.censor_text,
        scale_percent=round(detect_scale * 100),
        persist_passes=max(1, min(255, args.persist_frames)),
        smooth_percent=round(args.smooth_alpha * 100),
        stats=show_stats,
        on_stats=(lambda text: print(dim(f"native: {text}"), flush=True)) if show_stats else None,
    )
    server.start()

    # Spawn Swift helper
    print("Starting screen capture...", flush=True)
    helper = spawn_helper(socket_path, fps, display=args.display)
    stderr_tail = StderrTail(helper.stderr) if helper.stderr is not None else None

    print(bold("Running...") + " press Ctrl+C to stop.", flush=True)

    session = LiveSession(
        server,
        detector,
        censor_classes=censor_classes,
        padding=padding,
        full_censor=bool(args.full_censor),
        is_nudenet=is_nudenet,
        stats=stats,
        min_padding=min_padding_int,
        extra_factors=extra_factors,
    )
    # Watchdog: if our parent dies we are reparented to init (ppid 1).
    # Record the parent at start; only act when we started with a real
    # parent (ppid != 1). A terminal closing also reparents, and exiting
    # non-zero there is fine.
    parent_pid = os.getppid()
    # SIGTERM (menu-bar stop escalation, launchd, etc.) shuts down cleanly
    # with exit 0, like Ctrl+C.
    _prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(sig, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)
    helper_exit_code: int | None = None
    try:
        while True:
            try:
                if parent_pid != 1 and os.getppid() == 1:
                    print(
                        f"\n{error('Error:')} parent process died, shutting down",
                        file=sys.stderr,
                    )
                    helper_exit_code = 1
                    break
            except OSError:
                pass
            # Check if helper is still running
            if helper.poll() is not None:
                stderr_out = stderr_tail.text() if stderr_tail is not None else ""
                print(
                    f"\n{error('Error:')} Swift helper exited (code {helper.returncode})",
                    file=sys.stderr,
                )
                if stderr_out:
                    print(f"Helper stderr: {stderr_out}", file=sys.stderr)
                code = helper.returncode
                helper_exit_code = code if code else 1
                break

            try:
                session.step(0.1)
            except OSError as e:
                # Socket closed under us (helper exited); the poll above reports the exit.
                logger.debug("Live session send failed: %s", e)
                if helper.poll() is None:
                    print(f"\n{error('Error:')} lost connection to Swift helper", file=sys.stderr)
                    helper_exit_code = 1
                    break
                code = helper.returncode
                helper_exit_code = code if code else 1
                break

    except KeyboardInterrupt:
        print(f"\n{bold('Shutting down...')}")
    finally:
        try:
            signal.signal(signal.SIGTERM, _prev_sigterm)
        except OSError, ValueError:
            pass
        server.shutdown()
        detector.close()
        if show_stats and stats is not None:
            print(dim(stats.summary()), flush=True)
        if helper.poll() is None:
            helper.terminate()
            try:
                helper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                helper.kill()
                helper.wait()
        print(bold("Stopped."))
    if helper_exit_code:
        return helper_exit_code
    return 0


def _expected_output(input_path: str, model: str | None) -> str:
    """Compute the default output path for an input file."""
    from bsafe.detector import DEFAULT_MODEL

    stem, ext = os.path.splitext(input_path)
    tag = model or DEFAULT_MODEL
    return f"{stem}.{tag}.bsafe{ext}"


def _filter_inputs(
    inputs: list[str], supported_extensions: set[str], file_type: str, model: str | None
) -> tuple[list[str], int, int]:
    """Filter input paths for batch processing.

    Returns (to_process, skipped_existing, filtered_out).
    """
    filtered_out = 0
    skipped_existing = 0
    not_found = 0
    to_process = []

    for path in inputs:
        if os.path.isdir(path):
            filtered_out += 1
            continue
        if not os.path.isfile(path):
            print(f"{warn('Warning:')} file not found: {path}", file=sys.stderr)
            not_found += 1
            continue
        _, ext = os.path.splitext(path)
        if ext.lower() not in supported_extensions:
            filtered_out += 1
            continue
        if ".bsafe." in os.path.basename(path):
            filtered_out += 1
            continue
        expected = _expected_output(path, model)
        if os.path.exists(expected):
            skipped_existing += 1
            continue
        to_process.append(path)

    # Print summary
    if to_process:
        names = [os.path.basename(p) for p in to_process[:5]]
        listing = ", ".join(names)
        if len(to_process) > 5:
            listing += f", ... and {len(to_process) - 5} more"
        print(f"Found {info(str(len(to_process)))} {file_type}(s): {listing}")
    if filtered_out > 0:
        print(dim(f"Skipped {filtered_out} non-{file_type} file(s)"))
    if not_found > 0:
        print(dim(f"Skipped {not_found} missing file(s)"))
    if skipped_existing > 0:
        print(dim(f"Skipping {skipped_existing} file(s) with existing output"))
    if to_process:
        print(f"Processing {info(str(len(to_process)))} file(s)...")

    return to_process, skipped_existing, filtered_out


def _print_batch_summary(total: int, skipped: int, errors: list[tuple[str, str]]) -> None:
    """Print a summary after batch processing."""
    processed = total - len(errors)
    parts = [f"{success(str(processed))} processed"]
    if skipped:
        parts.append(f"{dim(str(skipped))} skipped")
    if errors:
        parts.append(f"{error(str(len(errors)))} failed")
    print(f"Batch complete: {', '.join(parts)}")
    if errors:
        for path, msg in errors:
            print(f"  {os.path.basename(path)}: {msg}", file=sys.stderr)


def _video_worker(
    error_queue,
    input_path: str,
    kwargs: dict,
) -> None:
    """Top-level function for subprocess video processing (must be picklable)."""
    try:
        from bsafe.video import process_video

        process_video(input_path, **kwargs)
    except Exception as e:
        error_queue.put(str(e))
        raise


def _run_file_subprocess(target, *args) -> tuple[int, str | None]:
    """Run target(*args) in a spawned subprocess. Returns (exit_code, error_message)."""
    ctx = multiprocessing.get_context("spawn")
    error_queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(error_queue, *args))
    proc.start()
    proc.join()
    error_msg = None
    if not error_queue.empty():
        error_msg = error_queue.get_nowait()
    exit_code = proc.exitcode if proc.exitcode is not None else 1
    return exit_code, error_msg


# Process tree architecture for batch mode:
#
# bsafe video *.mp4 --blur
#   main process (CLI, orchestration loop)
#     ├── file_1 subprocess (process_video → spawns chunk subprocesses)
#     │     ├── chunk_0 subprocess (loads model, processes frames)
#     │     ├── chunk_1 subprocess
#     │     └── ...
#     ├── file_2 subprocess
#     │     └── ...
#     └── gc.collect() between files
#
# bsafe image *.jpg --pixels
#   main process (CLI, orchestration loop)
#     ├── process_image(file_1) → gc.collect()  (detector created+closed internally)
#     ├── process_image(file_2) → gc.collect()
#     └── ...
#
# Video uses subprocess-per-file for full memory isolation (videos are heavy).
# Image processes in-process sequentially (images are lightweight).


def cmd_video(args):
    _validate_censor_args(args)
    _resolve_confidence(args)
    _warn_erax_unsupported(args)
    _print_config(args)

    from bsafe.censor import CensorConfig
    from bsafe.video import SUPPORTED_EXTENSIONS, process_video

    censor_config = CensorConfig(
        preset=args.censor,
        covered=args.covered,
        face_male=args.face_male,
        face_female=args.face_female,
        feet=args.feet,
        buttocks=args.buttocks,
    )

    inputs = args.input
    if args.output and len(inputs) > 1:
        print(
            f"{error('Error:')} -o/--output cannot be used with multiple input files",
            file=sys.stderr,
        )
        sys.exit(1)

    video_kwargs = dict(
        confidence=args.confidence,
        censor_config=censor_config,
        padding=args.padding,
        persist_frames=args.persist_frames,
        smooth_alpha=args.smooth_alpha,
        blur=args.blur,
        pixels=args.pixels,
        censor_text=args.censor_text,
        full_censor=args.full_censor,
        fps_override=args.fps,
        model=args.model,
        chunk_frames=args.chunk_frames,
        verbose=args.verbose,
        enhance=args.enhance,
    )

    if len(inputs) == 1:
        # Single-file mode: preserve existing behavior exactly
        try:
            process_video(inputs[0], output_path=args.output, **video_kwargs)
            print("\a", end="", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"\n{error('Error:')} {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode
    to_process, skipped, _ = _filter_inputs(inputs, SUPPORTED_EXTENSIONS, "video", model=args.model)
    if not to_process:
        print("Nothing to process.")
        return

    errors: list[tuple[str, str]] = []
    for i, path in enumerate(to_process, 1):
        print(f"\n{info(f'[{i}/{len(to_process)}]')} {os.path.basename(path)}")
        try:
            exit_code, error_msg = _run_file_subprocess(
                _video_worker,
                path,
                {**video_kwargs, "output_path": None},
            )
            if exit_code != 0:
                errors.append((path, error_msg or f"exit code {exit_code}"))
        except KeyboardInterrupt:
            print(f"\nInterrupted after {i - 1}/{len(to_process)} file(s).")
            sys.exit(130)
        gc.collect()

    _print_batch_summary(len(to_process), skipped, errors)
    print("\a", end="", flush=True)
    if errors:
        sys.exit(1)


def cmd_image(args):
    _validate_censor_args(args)
    _resolve_confidence(args)
    _warn_erax_unsupported(args)
    _print_config(args)

    from bsafe.censor import CensorConfig
    from bsafe.image import SUPPORTED_EXTENSIONS, process_image

    censor_config = CensorConfig(
        preset=args.censor,
        covered=args.covered,
        face_male=args.face_male,
        face_female=args.face_female,
        feet=args.feet,
        buttocks=args.buttocks,
    )

    inputs = args.input
    if args.output and len(inputs) > 1:
        print(
            f"{error('Error:')} -o/--output cannot be used with multiple input files",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(inputs) == 1:
        # Single-file mode: preserve existing behavior exactly
        try:
            output_path = process_image(
                inputs[0],
                output_path=args.output,
                confidence=args.confidence,
                censor_config=censor_config,
                padding=args.padding,
                blur=args.blur,
                pixels=args.pixels,
                censor_text=args.censor_text,
                full_censor=args.full_censor,
                model=args.model,
                verbose=args.verbose,
            )
            print(output_path)
            print("\a", end="", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"\n{error('Error:')} {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode
    to_process, skipped, _ = _filter_inputs(inputs, SUPPORTED_EXTENSIONS, "image", model=args.model)
    if not to_process:
        print("Nothing to process.")
        return

    errors: list[tuple[str, str]] = []
    for i, path in enumerate(to_process, 1):
        print(f"\n{info(f'[{i}/{len(to_process)}]')} {os.path.basename(path)}")
        try:
            output_path = process_image(
                path,
                confidence=args.confidence,
                censor_config=censor_config,
                padding=args.padding,
                blur=args.blur,
                pixels=args.pixels,
                censor_text=args.censor_text,
                full_censor=args.full_censor,
                model=args.model,
                verbose=args.verbose,
            )
            print(f"  → {output_path}")
        except KeyboardInterrupt:
            print(f"\nInterrupted after {i - 1}/{len(to_process)} file(s).")
            sys.exit(130)
        except Exception as e:
            print(f"  {error('Error:')} {e}", file=sys.stderr)
            errors.append((path, str(e)))
        gc.collect()

    _print_batch_summary(len(to_process), skipped, errors)
    print("\a", end="", flush=True)
    if errors:
        sys.exit(1)


def cmd_bench(args):
    """Dispatch the offline replay benchmark (bsafe bench detect|replay|sweep)."""
    from bsafe import bench

    handlers = {
        "detect": bench.cmd_detect,
        "replay": bench.cmd_replay,
        "sweep": bench.cmd_sweep,
    }
    handlers[args.bench_command](args)


def cmd_displays(args):
    from bsafe.swift_helper import list_displays

    try:
        displays = list_displays()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"{error('Error:')} {e}", file=sys.stderr)
        sys.exit(1)

    if not displays:
        print("No displays found.")
        return

    print(f"{'ID':<12} {'Resolution':<16} {'Role'}")
    print(dim(f"{'─' * 12} {'─' * 16} {'─' * 11}"))
    for d in displays:
        res = f"{d['width']}x{d['height']}"
        label = bold("(primary)") if d.get("primary") else "(secondary)"
        print(f"{d['id']:<12} {res:<16} {label}")


def cmd_bootstrap(args):
    import subprocess

    from bsafe.config import CONFIG_PATH, generate_default_config

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    swift_dir = os.path.join(project_dir, "swift")

    # Step 1: uv sync
    print("Installing Python dependencies...")
    result = subprocess.run(["uv", "sync"], cwd=project_dir)
    if result.returncode != 0:
        print(f"{error('Error:')} uv sync failed", file=sys.stderr)
        sys.exit(1)
    print(f"Python dependencies {success('OK')}\n")

    # Step 2: Build Swift helper
    print("Building Swift helper...")
    if not os.path.isdir(swift_dir):
        print(f"{error('Error:')} swift directory not found at {swift_dir}", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(["swift", "build", "-c", "release"], cwd=swift_dir)
    if result.returncode != 0:
        print(f"{error('Error:')} Swift build failed", file=sys.stderr)
        sys.exit(1)
    print(f"Swift helper {success('OK')}\n")

    # Step 3: Ensure models directory exists
    models_dir = os.path.expanduser("~/.config/bsafe/models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models directory: {models_dir} {success('OK')}\n")

    # Step 4: Create config file if it doesn't exist
    config_path = CONFIG_PATH
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            f.write(generate_default_config())
        print(f"Config file: {config_path} {success('CREATED')}\n")
    else:
        print(f"Config file: {config_path} already exists, skipping\n")

    print("Bootstrap complete! Run 'bsafe doctor' to verify.")
    _print_alias_hint()


def cmd_doctor(args):
    from bsafe.config import CONFIG_PATH

    all_ok = True

    # Python version
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {version}", end="")
    if sys.version_info >= (3, 14):
        print(f" {success('OK')}")
    else:
        print(f" {warn('WARN')}: expected >= 3.14")
        all_ok = False

    # macOS platform
    print(f"Platform: {sys.platform}", end="")
    if sys.platform == "darwin":
        print(f" {success('OK')}")
    else:
        print(f" {warn('WARN')}: bsafe requires macOS")
        all_ok = False

    # Config file (informational only)
    print(f"Config file: {CONFIG_PATH}", end="")
    if os.path.exists(CONFIG_PATH):
        print(f" {success('OK')}")
    else:
        print(f" {warn('NOT FOUND')} (optional — run bootstrap to create)")

    # NudeNet importable
    print("NudeNet: ", end="")
    try:
        import nudenet  # noqa: F401

        print(success("OK"))
    except ImportError:
        print(f"{warn('NOT FOUND')} — run: uv sync")
        all_ok = False

    # ONNX Runtime CoreML (live GPU detection; warn only)
    print("ONNX Runtime CoreML: ", end="")
    try:
        import onnxruntime

        if "CoreMLExecutionProvider" in onnxruntime.get_available_providers():
            print(success("OK"))
        else:
            print(f"{warn('WARN')} — CPU fallback (live detection will be slower)")
    except ImportError:
        print(f"{warn('WARN')} — CPU fallback (live detection will be slower)")

    # Swift helper binary
    print("Swift helper: ", end="")
    from bsafe.swift_helper import find_helper

    helper = find_helper()
    if helper:
        print(f"{success('OK')} ({helper})")
    else:
        print(f"{warn('NOT FOUND')} — build with: cd swift && swift build -c release")
        all_ok = False

    # Screen Recording permission
    print("Screen Recording: ", end="")
    if helper:
        import subprocess

        result = subprocess.run(
            [str(helper), "--check-permission"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "SCREEN_RECORDING_OK" in result.stdout:
            print(success("OK"))
        else:
            print(
                f"{warn('NOT GRANTED')} — open System Settings > Privacy & Security > Screen Recording "
                "and enable your terminal app"
            )
            all_ok = False
    else:
        print(f"{warn('SKIPPED')} (Swift helper not found)")
        all_ok = False

    if not all_ok:
        sys.exit(1)

    # Suggest shell alias
    _print_alias_hint()


def _print_alias_hint():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print()
    print("Hint: to make bsafe globally available, add this to your ~/.zshrc:")
    print(f"  alias bsafe='uv run --project {project_dir} bsafe'")


def _build_parser():
    """Construct the CLI argument parser. Returns (parser, start_parser, video_parser, image_parser)."""
    parser = argparse.ArgumentParser(prog="bsafe", description="Censor NSFW content on screen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start subcommand
    start_parser = subparsers.add_parser("start", help="Start the censoring process")
    start_parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Capture FPS; up to the display refresh rate, e.g. 120 on ProMotion (default: 60)",
    )
    _add_censor_args(start_parser)
    _add_temporal_args(start_parser)
    start_parser.add_argument(
        "--display",
        default="primary",
        help="Display to capture: primary, secondary, all, or numeric ID (default: primary)",
    )
    start_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run without Swift helper or detector",
    )
    start_parser.add_argument(
        "--stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log live detection timing every ~2s plus native capture/tracking stats",
    )
    start_parser.add_argument(
        "--detect-scale",
        type=float,
        default=1.0,
        help="Detection resolution as a multiple of the display's point size (default 1.0). "
        "1.5 detects images down to ~100 pt wide at ~2x GPU cost",
    )
    start_parser.add_argument(
        "--min-padding",
        type=int,
        default=0,
        help="Minimum padding in capture pixels per side, added to small boxes (default: 0; try 24)",
    )
    start_parser.add_argument(
        "--extra-scales",
        type=str,
        default=None,
        help="Extra detection passes at these multiples of point size (comma list, each < --detect-scale; "
        "default 0.5 for NudeNet, 'none' to disable). Catches large close-ups the full-res pass misses",
    )

    # video subcommand
    video_parser = subparsers.add_parser(
        "video", help="Process a video file and output a censored copy"
    )
    video_parser.add_argument("input", nargs="+", help="Path(s) to video file(s)")
    video_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: <input>.<model>.bsafe.<ext>)",
    )
    video_parser.add_argument(
        "--fps", type=int, default=None, help="Detection FPS override (default: native)"
    )
    video_parser.add_argument(
        "--chunk-frames",
        type=int,
        default=None,
        help="Frames per processing chunk (default: 5000). "
        "Smaller chunks use less memory but add brief tracking gaps at boundaries.",
    )
    video_parser.add_argument(
        "--enhance",
        choices=["dim"],
        default=None,
        help="enhancement mode: 'dim' for low-light footage (default: disabled)",
    )
    _add_censor_args(video_parser)
    _add_temporal_args(video_parser)

    # image subcommand
    image_parser = subparsers.add_parser(
        "image", help="Process an image file and output a censored copy"
    )
    image_parser.add_argument("input", nargs="+", help="Path(s) to image file(s)")
    image_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: <input>.<model>.bsafe.<ext>)",
    )
    _add_censor_args(image_parser)

    # bench subcommand (offline replay benchmark)
    bench_parser = subparsers.add_parser("bench", help="Offline replay benchmark")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_command", required=True)

    bench_detect_parser = bench_subparsers.add_parser(
        "detect", help="Detect on every frame of a recording (writes DIR/dets.jsonl)"
    )
    bench_detect_parser.add_argument("input", help="Path to the screen recording")
    bench_detect_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory; dets.jsonl is written there",
    )
    bench_detect_parser.add_argument(
        "--model",
        choices=["320n", "640m"],
        default="320n",
        help="NudeNet model: '320n' (default, bundled) or '640m' (requires download)",
    )
    bench_detect_parser.add_argument(
        "--extra-scales",
        type=str,
        default=None,
        help="Extra detection passes as frame-relative factors, e.g. '0.5' (the recording "
        "is already at the live capture size, so factors are relative to the frame)",
    )

    bench_replay_parser = bench_subparsers.add_parser(
        "replay", help="Replay detections through the Swift tracker and score the result"
    )
    bench_replay_parser.add_argument("dir", help="Directory produced by 'bsafe bench detect'")
    bench_replay_parser.add_argument(
        "--censor",
        choices=["none", "female", "male", "all", "body"],
        default="body",
        help="What to censor: none, female, male, all, or body (default: body)",
    )
    bench_replay_parser.add_argument(
        "--padding",
        type=float,
        default=0.4,
        help="Box expansion fraction (default: 0.4)",
    )
    bench_replay_parser.add_argument(
        "--min-padding",
        type=int,
        default=0,
        help="Minimum padding in capture pixels per side (added to small boxes)",
    )
    bench_replay_parser.add_argument(
        "--persist-passes",
        type=int,
        default=8,
        help="Tracker passes a box persists after disappearing (default: 8)",
    )
    bench_replay_parser.add_argument(
        "--min-persist-s",
        type=float,
        default=1.0,
        help="Minimum persistence time in seconds (default: 1.0)",
    )
    bench_replay_parser.add_argument(
        "--shrink-alpha",
        type=float,
        default=0.1,
        help="Per-0.1s shrink factor for inward edges (default: 0.1)",
    )
    bench_replay_parser.add_argument(
        "--present-lead-ms",
        type=float,
        default=16.0,
        help="How far ahead of a frame's pts its boxes are shown (default: 16)",
    )
    bench_replay_parser.add_argument(
        "--overhead-ms",
        type=float,
        default=6.0,
        help="Simulated detection/IPC overhead per request (default: 6)",
    )
    bench_replay_parser.add_argument(
        "--name",
        default=None,
        help="Run subdirectory name (default: derived from the tracker config)",
    )
    bench_replay_parser.add_argument(
        "--gt-dets",
        action="append",
        default=None,
        help="GT dets.jsonl for recall scoring (repeatable; default: DIR/dets.jsonl)",
    )
    bench_replay_parser.add_argument(
        "--gt-conf",
        type=float,
        default=0.25,
        help="Min confidence for all GT sources (default: 0.25)",
    )

    bench_sweep_parser = bench_subparsers.add_parser(
        "sweep", help="Run a grid of padding/tracker configs and print a table"
    )
    bench_sweep_parser.add_argument("dir", help="Directory produced by 'bsafe bench detect'")
    bench_sweep_parser.add_argument(
        "--censor",
        choices=["none", "female", "male", "all", "body"],
        default="body",
        help="What to censor: none, female, male, all, or body (default: body)",
    )
    bench_sweep_parser.add_argument(
        "--min-padding",
        type=int,
        default=0,
        help="Minimum padding in capture pixels per side, applied to all sweep configs",
    )
    bench_sweep_parser.add_argument(
        "--gt-dets",
        action="append",
        default=None,
        help="GT dets.jsonl for recall scoring (repeatable; default: DIR/dets.jsonl)",
    )
    bench_sweep_parser.add_argument(
        "--gt-conf",
        type=float,
        default=0.25,
        help="Min confidence for all GT sources (default: 0.25)",
    )

    subparsers.add_parser("displays", help="List connected displays")
    subparsers.add_parser("doctor", help="Check system requirements")
    subparsers.add_parser("bootstrap", help="Install dependencies and build Swift helper")

    return parser, start_parser, video_parser, image_parser


class _DetectionColorFilter(logging.Filter):
    """Color [detection] prefixes magenta in verbose log output."""

    def filter(self, record):
        from bsafe.style import detection

        if record.msg and "[detection]" in str(record.msg):
            record.msg = record.msg.replace("[detection]", detection("[detection]"), 1)
        return True


def _add_detection_filter():
    """Add detection coloring filter to the bsafe logger."""
    bsafe_logger = logging.getLogger("bsafe")
    bsafe_logger.addFilter(_DetectionColorFilter())


def main():
    from bsafe.config import load_config

    parser, start_parser, video_parser, image_parser = _build_parser()

    # Load user config and apply as defaults to subparsers
    config = load_config()
    if config:
        start_config = {**config.get("common", {}), **config.get("start", {})}
        video_config = {**config.get("common", {}), **config.get("video", {})}
        image_config = {**config.get("common", {}), **config.get("image", {})}
        if start_config:
            start_parser.set_defaults(**start_config)
        if video_config:
            video_parser.set_defaults(**video_config)
        if image_config:
            image_parser.set_defaults(**image_config)

    args = parser.parse_args()

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)
        _add_detection_filter()
    else:
        logging.basicConfig(level=logging.WARNING)

    commands = {
        "start": cmd_start,
        "displays": cmd_displays,
        "doctor": cmd_doctor,
        "bootstrap": cmd_bootstrap,
        "video": cmd_video,
        "image": cmd_image,
        "bench": cmd_bench,
    }

    faulthandler.enable()
    result = commands[args.command](args)
    if isinstance(result, int) and result != 0:
        sys.exit(result)
