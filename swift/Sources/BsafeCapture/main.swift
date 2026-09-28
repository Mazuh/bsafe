import AppKit
import BsafeCore
import Foundation
import ScreenCaptureKit

// MARK: - Argument parsing

func parseArgs() -> (socketPath: String, fps: Int, display: String?) {
    let args = CommandLine.arguments
    var socketPath = ""
    var fps = 3
    var display: String? = nil

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--socket":
            i += 1
            guard i < args.count else {
                fputs("Error: --socket requires a path\n", stderr)
                exit(1)
            }
            socketPath = args[i]
        case "--fps":
            i += 1
            guard i < args.count, let f = Int(args[i]), f > 0 else {
                fputs("Error: --fps requires a positive integer\n", stderr)
                exit(1)
            }
            fps = f
        case "--display":
            i += 1
            guard i < args.count else {
                fputs("Error: --display requires a value\n", stderr)
                exit(1)
            }
            display = args[i]
        default:
            fputs("Unknown argument: \(args[i])\n", stderr)
            exit(1)
        }
        i += 1
    }

    guard !socketPath.isEmpty else {
        fputs("Error: --socket is required\n", stderr)
        exit(1)
    }

    return (socketPath, fps, display)
}

// MARK: - Big-endian packing helpers

func appendU16(_ data: inout Data, _ value: UInt16) {
    var be = value.bigEndian
    withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
}

func appendU32(_ data: inout Data, _ value: UInt32) {
    var be = value.bigEndian
    withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
}

// MARK: - CMD_START parsing

struct StartParams {
    var fps: Int
    var scalePercent: Int
    var persistPasses: Int
    var smoothPercent: Int
    var stats: Bool
}

/// Parse CMD_START payload: `!BHBBB` (fps, scale_percent, persist, smooth, flags)
/// or legacy 1-byte (fps only). --fps stays the pre-CMD_START fallback.
func parseStart(_ payload: Data, fallbackFps: Int) -> StartParams {
    if payload.count >= 6 {
        let fpsRaw = Int(payload[0])
        let scaleRaw = (Int(payload[1]) << 8) | Int(payload[2])
        let persistRaw = Int(payload[3])
        let smoothRaw = Int(payload[4])
        let flags = payload[5]
        let fps = fpsRaw >= 1 ? fpsRaw : fallbackFps
        var scale = scaleRaw
        if scale < 100 { scale = 100 }
        if scale > 200 { scale = 200 }
        var persist = persistRaw
        if persist < 1 { persist = 1 }
        if persist > 255 { persist = 255 }
        var smooth = smoothRaw
        if smooth < 0 { smooth = 0 }
        if smooth > 100 { smooth = 100 }
        return StartParams(
            fps: fps, scalePercent: scale, persistPasses: persist,
            smoothPercent: smooth, stats: (flags & 0x01) != 0
        )
    } else if payload.count == 1 {
        let f = Int(payload[0])
        return StartParams(
            fps: f >= 1 ? f : fallbackFps, scalePercent: 100,
            persistPasses: 8, smoothPercent: 50, stats: false
        )
    } else {
        return StartParams(
            fps: fallbackFps, scalePercent: 100,
            persistPasses: 8, smoothPercent: 50, stats: false
        )
    }
}

// MARK: - CMD_CENSOR_SEQ parsing

/// Parse CMD_CENSOR_SEQ payload: `!IIIIH` (display_id, frame_w, frame_h, seq, box_count)
/// then box_count x `!iiii` (x, y, w, h) in capture pixels of frame seq.
func parseCensorSeq(_ data: Data) -> (displayID: UInt32, frameW: Int, frameH: Int, seq: UInt32, boxes: [TrackBox])? {
    let headerSize = 18  // 4 + 4 + 4 + 4 + 2
    guard data.count >= headerSize else { return nil }

    let displayID = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self).bigEndian }
    let frameW = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self).bigEndian }
    let frameH = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self).bigEndian }
    let seq = data.withUnsafeBytes { $0.load(fromByteOffset: 12, as: UInt32.self).bigEndian }
    let boxCount = (Int(data[16]) << 8) | Int(data[17])

    guard data.count >= headerSize + Int(boxCount) * 16 else { return nil }

    var boxes: [TrackBox] = []
    boxes.reserveCapacity(Int(boxCount))
    for i in 0..<Int(boxCount) {
        let offset = headerSize + i * 16
        let x = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Int32.self).bigEndian }
        let y = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Int32.self).bigEndian }
        let w = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Int32.self).bigEndian }
        let h = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 12, as: Int32.self).bigEndian }
        guard w > 0, h > 0 else { continue }
        boxes.append(TrackBox(x: Double(x), y: Double(y), w: Double(w), h: Double(h)))
    }

    return (displayID, Int(frameW), Int(frameH), seq, boxes)
}

// MARK: - Permission check (early exit)

if CommandLine.arguments.contains("--check-permission") {
    let semaphore = DispatchSemaphore(value: 0)
    var ok = false

    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, error in
        if error == nil, let content, !content.displays.isEmpty {
            ok = true
        }
        semaphore.signal()
    }

    semaphore.wait()
    print(ok ? "SCREEN_RECORDING_OK" : "SCREEN_RECORDING_DENIED")
    exit(ok ? 0 : 1)
}

// MARK: - List displays (early exit)

if CommandLine.arguments.contains("--list-displays") {
    // Consume --list-displays and reject unknown flags
    let knownFlags: Set<String> = ["--list-displays"]
    for arg in CommandLine.arguments.dropFirst() {
        if !knownFlags.contains(arg) {
            fputs("Error: unexpected argument with --list-displays: \(arg)\n", stderr)
            exit(1)
        }
    }

    let semaphore = DispatchSemaphore(value: 0)

    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, error in
        if let error {
            fputs("Error: \(error.localizedDescription)\n", stderr)
            semaphore.signal()
            return
        }

        guard let content else {
            fputs("Error: no shareable content\n", stderr)
            semaphore.signal()
            return
        }

        let primaryID = CGMainDisplayID()
        for display in content.displays {
            let isPrimary = display.displayID == primaryID
            let entry: [String: Any] = [
                "id": display.displayID,
                "width": display.width,
                "height": display.height,
                "primary": isPrimary,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: entry),
               let json = String(data: data, encoding: .utf8) {
                print(json)
            }
        }
        semaphore.signal()
    }

    semaphore.wait()
    exit(0)
}

// MARK: - Main

let (socketPath, requestedFps, displayArg) = parseArgs()

print("BsafeCapture: connecting to \(socketPath), requested FPS=\(requestedFps), display=\(displayArg ?? "primary")")

let client = SocketClient(path: socketPath)

do {
    try client.connect()
} catch {
    fputs("Failed to connect to socket: \(error)\n", stderr)
    exit(1)
}

// Read CMD_START from Python
let cmdStart: (type: UInt8, payload: Data)
do {
    cmdStart = try client.readMessage()
} catch {
    fputs("Failed to read CMD_START: \(error)\n", stderr)
    exit(1)
}
guard cmdStart.type == 0x10 else {
    fputs("Expected CMD_START (0x10), got 0x\(String(cmdStart.type, radix: 16))\n", stderr)
    exit(1)
}

let params = parseStart(cmdStart.payload, fallbackFps: requestedFps)
let captureScale = Double(params.scalePercent) / 100.0
print("BsafeCapture: server start — fps=\(params.fps) scale=\(params.scalePercent)% persist=\(params.persistPasses) stats=\(params.stats ? "on" : "off")")

// Initialize NSApplication for overlay windows (no dock icon)
let app = NSApplication.shared
app.setActivationPolicy(.accessory)

// Censor style config (set by CMD_CENSOR_STYLE from Python).
// Protected by censorStyleLock — written on the reader thread, snapshotted by pipelines.
var censorBlur: Double = 0.0
var censorPixels: Double = 0.0
var censorText: String? = nil
let censorStyleLock = NSLock()
let styleProvider: () -> CensorStyle = {
    censorStyleLock.lock()
    defer { censorStyleLock.unlock() }
    return CensorStyle(blur: censorBlur, pixels: censorPixels, text: censorText)
}

// Frame send path: one global serial queue; sends never block the display queues.
// A failed write can leave a partial framed message on the stream, after which
// every later message is corrupt — so a send failure is fatal.
let frameSendQueue = DispatchQueue(label: "bsafe.send")
let frameSender: (UInt32, UInt32, UInt32, Int64, UInt32, Data) -> Void = { did, w, h, ptsNs, seq, pixels in
    frameSendQueue.async {
        do {
            try client.sendFrameRaw(displayID: did, width: w, height: h, ptsNs: ptsNs, seq: seq, pixels: pixels)
        } catch {
            fputs("BsafeCapture: failed to send frame: \(error)\n", stderr)
            exit(1)
        }
    }
}

// Resolve displays with the same filter semantics as before.
let resolved: (displays: [SCDisplay], content: SCShareableContent)
do {
    resolved = try resolveDisplays(filter: displayArg)
} catch {
    fputs("Failed to resolve displays: \(error)\n", stderr)
    exit(1)
}

let capturable = isOverlayCapturable()
if capturable {
    print("BsafeCapture: BSAFE_OVERLAY_CAPTURABLE=1 — overlay visible to recorders, excluded from own capture")
}

let trackerConfig = TrackerConfig(
    persistPasses: params.persistPasses
)
// smoothPercent is parsed for protocol compatibility; the live tracker no longer uses it.
let leadSeconds = presentLeadSeconds()

// Create one overlay (synchronously on the main thread) + pipeline per
// display, start its stream, then report MSG_DISPLAY_INFO before any frame.
var started: [(displayID: UInt32, pipeline: DisplayPipeline)] = []
var startupIDs: [String] = []
var overlaysByDisplay: [UInt32: CensorOverlay] = [:]
for scDisplay in resolved.displays {
    let overlay = CensorOverlay()
    overlay.setup(displayID: scDisplay.displayID, capturable: capturable)
    overlaysByDisplay[scDisplay.displayID] = overlay
}
// Reusable exclusion computation: restarts recompute this via the provider
// below (windows can be re-created after sleep/lock). When the overlay is not
// capturable this returns [] without touching NSWindow; otherwise window ids
// are snapshotted on the main thread (startup runs on main, restarts run on a
// background queue — snapshotOverlayWindowIDs handles both).
let excludedWindowsProvider: () -> [SCWindow] = {
    guard capturable else { return [] }
    let ids = snapshotOverlayWindowIDs(Array(overlaysByDisplay.values))
    return computeExcludedWindows(capturable: capturable, overlayWindowIDs: ids)
}
let excludedWindows = excludedWindowsProvider()
if capturable {
    let ids = snapshotOverlayWindowIDs(Array(overlaysByDisplay.values))
    if excludedWindows.count != ids.count {
        fputs("BsafeCapture: warning: could only exclude \(excludedWindows.count)/\(ids.count) overlay windows from capture\n", stderr)
    }
}
for scDisplay in resolved.displays {
    let did: UInt32 = scDisplay.displayID
    let pointsW = scDisplay.width
    let pointsH = scDisplay.height
    let capW = Int((Double(pointsW) * captureScale).rounded())
    let capH = Int((Double(pointsH) * captureScale).rounded())

    guard let overlay = overlaysByDisplay[did] else { continue }
    let pipeline = DisplayPipeline(
        displayID: did,
        captureWidth: capW,
        captureHeight: capH,
        pointsWidth: pointsW,
        pointsHeight: pointsH,
        scaleFactor: captureScale,
        presentLead: leadSeconds,
        config: trackerConfig,
        overlay: overlay,
        statsEnabled: params.stats,
        styleProvider: styleProvider,
        frameSender: frameSender
    )
    pipeline.configureRestart(fps: params.fps, excludedWindowsProvider: excludedWindowsProvider)
    do {
        try pipeline.startStream(display: scDisplay, fps: params.fps, excludingWindows: excludedWindows)
    } catch {
        fputs("BsafeCapture: failed to start capture for display \(did): \(error)\n", stderr)
        continue
    }

    var info = Data()
    appendU32(&info, did)
    appendU32(&info, UInt32(capW))
    appendU32(&info, UInt32(capH))
    appendU32(&info, UInt32(pointsW))
    appendU32(&info, UInt32(pointsH))
    do {
        try client.sendMessage(type: 0x03, payload: info)
    } catch {
        fputs("BsafeCapture: failed to send display info for display \(did): \(error)\n", stderr)
        exit(1)
    }

    started.append((displayID: did, pipeline: pipeline))
    startupIDs.append(String(did))
}

guard !started.isEmpty else {
    fputs("BsafeCapture: failed to start capture on any display\n", stderr)
    exit(1)
}

print("BsafeCapture: capturing \(started.count) display(s): [\(startupIDs.joined(separator: ", "))] at \(params.fps) FPS, scale=\(params.scalePercent)%")

let routes: [UInt32: DisplayPipeline] = Dictionary(uniqueKeysWithValues: started.map { ($0.displayID, $0.pipeline) })

// Debug hook (harmless): `kill -USR1 <BsafeCapture pid>` simulates a stream
// interruption on every display — stopCapture plus the same restart path as
// didStopWithError with synthetic code -1.
signal(SIGUSR1, SIG_IGN)
let usr1Source = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: DispatchQueue.global())
usr1Source.setEventHandler {
    for pipeline in routes.values {
        pipeline.simulateInterruption()
    }
}
usr1Source.resume()

// Stats timer (only when the server asked for native stats).
var statsTimer: DispatchSourceTimer? = nil
if params.stats {
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
    timer.schedule(deadline: .now() + 2.0, repeating: 2.0)
    timer.setEventHandler {
        for pipeline in routes.values {
            if let line = pipeline.takeStatsLine() {
                do {
                    try client.sendMessage(type: 0x04, payload: Data(line.utf8))
                } catch {
                    fputs("BsafeCapture: failed to send stats: \(error)\n", stderr)
                    exit(1)
                }
            }
        }
    }
    timer.resume()
    statsTimer = timer
}

// Listen for commands from Python in background
DispatchQueue.global().async {
    while true {
        do {
            let (msgType, payload) = try client.readMessage()
            switch msgType {
            case 0x12:  // CMD_REQUEST_FRAME: grant one frame credit
                guard payload.count >= 4 else { continue }
                let did = payload.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self).bigEndian }
                if let pipe = routes[did] {
                    pipe.queue.async { pipe.handleCredit() }
                }
            case 0x22:  // CMD_CENSOR_SEQ: detections for frame seq
                guard let det = parseCensorSeq(payload) else { continue }
                if let pipe = routes[det.displayID] {
                    pipe.queue.async {
                        pipe.handleDetections(
                            frameWidth: det.frameW, frameHeight: det.frameH,
                            seq: det.seq, boxes: det.boxes
                        )
                    }
                }
            case 0x21:  // CMD_CENSOR_STYLE
                guard payload.count >= 6 else {
                    fputs("BsafeCapture: CMD_CENSOR_STYLE payload too short\n", stderr)
                    continue
                }
                let blurPct = payload.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt16.self).bigEndian }
                let pixelsPct = payload.withUnsafeBytes { $0.load(fromByteOffset: 2, as: UInt16.self).bigEndian }
                let textLength = payload.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt16.self).bigEndian }
                censorStyleLock.lock()
                censorBlur = Double(blurPct) / 100.0
                censorPixels = Double(pixelsPct) / 100.0
                if textLength > 0, payload.count >= 6 + Int(textLength) {
                    censorText = String(data: payload[6..<(6 + Int(textLength))], encoding: .utf8)
                } else {
                    censorText = nil
                }
                censorStyleLock.unlock()
                print("BsafeCapture: censor style updated — blur=\(censorBlur), pixels=\(censorPixels), text=\(censorText ?? "none")")
            case 0x20:  // Legacy CMD_CENSOR: unused by live v2
                continue
            case 0xFF:  // CMD_SHUTDOWN
                print("BsafeCapture: received shutdown command")
                for pipeline in routes.values {
                    pipeline.stopStream()
                }
                client.disconnect()
                exit(0)
            default:
                break
            }
        } catch {
            // Connection lost
            print("BsafeCapture: connection lost, shutting down")
            for pipeline in routes.values {
                pipeline.stopStream()
            }
            exit(0)
        }
    }
}

// Run the app event loop (needed for NSWindow overlay rendering)
app.run()
