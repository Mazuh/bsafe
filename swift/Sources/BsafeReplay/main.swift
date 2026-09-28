import AVFoundation
import BsafeCore
import CoreMedia
import CoreVideo
import Foundation

func usage() -> String {
    return """
    Usage: bsafe-replay --video PATH --boxes PATH --out PATH [options]
      --video PATH          input screen recording
      --boxes PATH          input boxes.jsonl (header + per-frame boxes)
      --out PATH            output displayed.jsonl
      --present-lead-ms MS  default 16
      --overhead-ms MS      default 6
      --persist-passes N    default 8
      --shrink-alpha A      default 0.25
      --min-persist-s S     default 0.6
      --max-lead-s S        default 0.1
    """
}

func usageError(_ msg: String) -> Never {
    fputs("Error: \(msg)\n", stderr)
    exit(2)
}

func ioError(_ msg: String) -> Never {
    // Keep to a single line for the coordinator.
    let oneLine = msg.split(whereSeparator: \.isNewline).joined(separator: " ")
    fputs("Error: \(oneLine)\n", stderr)
    exit(1)
}

func warn(_ msg: String) {
    let oneLine = msg.split(whereSeparator: \.isNewline).joined(separator: " ")
    fputs("Warning: \(oneLine)\n", stderr)
}

// MARK: - Flag parsing (no dependencies)

var videoPath: String?
var boxesPath: String?
var outPath: String?
var presentLeadMs = 16.0
var overheadMs = 6.0
var persistPasses = 8
var shrinkAlpha = 0.1
var minPersistS = 1.0
var maxLeadS = 0.1

let args = CommandLine.arguments
var i = 1
while i < args.count {
    let a = args[i]
    func needValue(_ flag: String) -> String {
        i += 1
        guard i < args.count else { usageError("\(flag) requires a value") }
        return args[i]
    }
    switch a {
    case "--video":
        videoPath = needValue(a)
    case "--boxes":
        boxesPath = needValue(a)
    case "--out":
        outPath = needValue(a)
    case "--present-lead-ms":
        let v = needValue(a)
        guard let d = Double(v) else { usageError("--present-lead-ms requires a number") }
        presentLeadMs = d
    case "--overhead-ms":
        let v = needValue(a)
        guard let d = Double(v) else { usageError("--overhead-ms requires a number") }
        overheadMs = d
    case "--persist-passes":
        let v = needValue(a)
        guard let n = Int(v) else { usageError("--persist-passes requires an integer") }
        persistPasses = n
    case "--shrink-alpha":
        let v = needValue(a)
        guard let d = Double(v) else { usageError("--shrink-alpha requires a number") }
        shrinkAlpha = d
    case "--min-persist-s":
        let v = needValue(a)
        guard let d = Double(v) else { usageError("--min-persist-s requires a number") }
        minPersistS = d
    case "--max-lead-s":
        let v = needValue(a)
        guard let d = Double(v) else { usageError("--max-lead-s requires a number") }
        maxLeadS = d
    case "-h", "--help":
        print(usage())
        exit(0)
    default:
        usageError("unknown argument: \(a)")
    }
    i += 1
}

guard let videoPath, !videoPath.isEmpty else { usageError("--video is required") }
guard let boxesPath, !boxesPath.isEmpty else { usageError("--boxes is required") }
guard let outPath, !outPath.isEmpty else { usageError("--out is required") }
if persistPasses < 1 { usageError("--persist-passes must be >= 1") }
if presentLeadMs.isNaN || presentLeadMs < 0 { usageError("--present-lead-ms must be >= 0") }
if overheadMs.isNaN || overheadMs < 0 { usageError("--overhead-ms must be >= 0") }
if shrinkAlpha.isNaN || shrinkAlpha < 0 || shrinkAlpha > 1 { usageError("--shrink-alpha must be in [0, 1]") }
if minPersistS.isNaN || minPersistS < 0 { usageError("--min-persist-s must be >= 0") }
if maxLeadS.isNaN || maxLeadS < 0 { usageError("--max-lead-s must be >= 0") }

let presentLead = presentLeadMs / 1000.0
let overheadS = overheadMs / 1000.0

// MARK: - Read boxes.jsonl with JSONSerialization

struct BoxFrame {
    var pts: Double
    var detectS: Double
    var boxes: [TrackBox]
}

func numToDouble(_ v: Any?) -> Double? {
    if let d = v as? Double { return d }
    if let n = v as? NSNumber { return n.doubleValue }
    if let n = v as? Int { return Double(n) }
    return nil
}

func numToInt(_ v: Any?) -> Int? {
    if let n = v as? Int { return n }
    if let n = v as? NSNumber { return n.intValue }
    if let d = v as? Double { return Int(d) }
    return nil
}

let boxesData: Data
do {
    boxesData = try Data(contentsOf: URL(fileURLWithPath: boxesPath))
} catch {
    ioError("cannot read boxes file: \(boxesPath)")
}

guard let boxesText = String(data: boxesData, encoding: .utf8) else {
    ioError("boxes file is not UTF-8: \(boxesPath)")
}

var rawLines = boxesText.split(separator: "\n")
    .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
    .filter { !$0.isEmpty }
guard !rawLines.isEmpty else {
    ioError("boxes file is empty: \(boxesPath)")
}

func parseJSONLine(_ line: String) -> [String: Any]? {
    guard let d = line.data(using: .utf8) else { return nil }
    return (try? JSONSerialization.jsonObject(with: d, options: [])) as? [String: Any]
}

guard let headerObj = parseJSONLine(rawLines[0]),
    let headerW = numToInt(headerObj["width"]),
    let headerH = numToInt(headerObj["height"]),
    let headerN = numToInt(headerObj["frames"])
else {
    ioError("invalid boxes header line")
}
if headerW <= 0 || headerH <= 0 || headerN < 0 {
    ioError("invalid boxes header dimensions")
}

var frames: [BoxFrame] = []
frames.reserveCapacity(max(0, rawLines.count - 1))
for (idx, line) in rawLines.dropFirst().enumerated() {
    guard let obj = parseJSONLine(line) else {
        ioError("invalid boxes JSON on line \(idx + 2)")
    }
    guard let fi = numToInt(obj["frame"]),
        let pts = numToDouble(obj["pts"]),
        let ds = numToDouble(obj["detect_s"])
    else {
        ioError("invalid boxes frame line \(idx + 2)")
    }
    guard let boxArr = obj["boxes"] as? [Any] else {
        ioError("invalid boxes field on line \(idx + 2)")
    }
    var tbs: [TrackBox] = []
    tbs.reserveCapacity(boxArr.count)
    for b in boxArr {
        guard let nums = b as? [Any], nums.count == 4,
            let x = numToDouble(nums[0]), let y = numToDouble(nums[1]),
            let w = numToDouble(nums[2]), let h = numToDouble(nums[3])
        else {
            ioError("invalid box on line \(idx + 2)")
        }
        tbs.append(TrackBox(x: x, y: y, w: w, h: h))
    }
    if fi != frames.count {
        ioError("boxes frame index out of order on line \(idx + 2): got \(fi), expected \(frames.count)")
    }
    frames.append(BoxFrame(pts: pts, detectS: ds, boxes: tbs))
}

let nBoxLines = frames.count
var nUse = min(headerN, nBoxLines)
if headerN != nBoxLines {
    if abs(headerN - nBoxLines) > 2 {
        ioError("boxes header frames=\(headerN) but found \(nBoxLines) frame lines")
    } else {
        warn("boxes header frames=\(headerN) but found \(nBoxLines) frame lines; using \(nUse)")
    }
}
if frames.count > nUse {
    frames = Array(frames.prefix(nUse))
}
if nUse == 0 {
    ioError("no frames to replay")
}

// MARK: - Open video with AVAssetReader (BGRA, native size)

let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
guard let videoTrack = asset.tracks(withMediaType: .video).first else {
    ioError("no video track in \(videoPath)")
}

let reader: AVAssetReader
do {
    reader = try AVAssetReader(asset: asset)
} catch {
    ioError("cannot open video \(videoPath): \(error.localizedDescription)")
}

let outputSettings: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
]
let trackOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
trackOutput.alwaysCopiesSampleData = false
guard reader.canAdd(trackOutput) else {
    ioError("cannot add reader output for \(videoPath)")
}
reader.add(trackOutput)
guard reader.startReading() else {
    ioError("cannot start reading \(videoPath)")
}

// MARK: - Tracker + scheduler

var cfg = TrackerConfig(
    persistPasses: persistPasses,
    shrinkAlpha: shrinkAlpha,
    minPersistSeconds: minPersistS
)
cfg.maxLeadSeconds = maxLeadS
let tracker = DisplayTracker(frameWidth: headerW, frameHeight: headerH, config: cfg)
var scheduler = ReplayScheduler(detectS: frames.map { $0.detectS }, overheadS: overheadS)

// MARK: - Buffered output

FileManager.default.createFile(atPath: outPath, contents: nil)
guard let outHandle = FileHandle(forWritingAtPath: outPath) else {
    ioError("cannot write output file: \(outPath)")
}
var outBuffer = ""
outBuffer.reserveCapacity(1 << 20)
var bufferedLines = 0
func flushBuffer() {
    if !outBuffer.isEmpty {
        guard let d = outBuffer.data(using: .utf8) else {
            ioError("cannot encode output")
        }
        outHandle.write(d)
        outBuffer = ""
        outBuffer.reserveCapacity(1 << 20)
        bufferedLines = 0
    }
}
func emitLine(_ s: String) {
    outBuffer += s
    outBuffer += "\n"
    bufferedLines += 1
    if bufferedLines >= 200 {
        flushBuffer()
    }
}

func boxesJSON(_ boxes: [TrackBox]) -> String {
    if boxes.isEmpty { return "[]" }
    var parts: [String] = []
    parts.reserveCapacity(boxes.count)
    for b in boxes {
        parts.append("[\(b.x),\(b.y),\(b.w),\(b.h)]")
    }
    return "[" + parts.joined(separator: ",") + "]"
}

// MARK: - Replay loop (pts from boxes.jsonl so Python and Swift agree)

var applied = 0
var ignored = 0
var processed = 0
var videoTotal = 0
var sizeChecked = false

while let sample = trackOutput.copyNextSampleBuffer() {
    guard let imgBuf = CMSampleBufferGetImageBuffer(sample) else {
        ioError("failed to decode frame \(videoTotal)")
    }
    let w = CVPixelBufferGetWidth(imgBuf)
    let h = CVPixelBufferGetHeight(imgBuf)
    if !sizeChecked {
        sizeChecked = true
        if w != headerW || h != headerH {
            ioError("decoded size \(w)x\(h) != header \(headerW)x\(headerH)")
        }
    } else if w != headerW || h != headerH {
        ioError("decoded size changed at frame \(videoTotal): \(w)x\(h)")
    }
    if videoTotal < nUse {
        let k = videoTotal
        let ptsK = frames[k].pts
        let completions = scheduler.beginFrameTimed(k: k, pts: ptsK)
        var lastOk: Int?
        for (detIdx, compTime) in completions {
            guard detIdx >= 0 && detIdx < frames.count else {
                ignored += 1
                continue
            }
            let ok = tracker.applyDetections(seq: UInt32(detIdx), boxes: frames[detIdx].boxes)
            if ok {
                applied += 1
                lastOk = detIdx
                // Live parity: re-render immediately at completion time, before
                // ingesting frame k. What is on screen at c uses now = c.
                let renderedAtCompletion = tracker.renderBoxes(now: compTime, presentLead: presentLead)
                let evFrame = k > 0 ? k - 1 : k
                emitLine("{\"frame\":\(evFrame),\"pts\":\(compTime),\"display_pts\":\(compTime + presentLead),\"boxes\":\(boxesJSON(renderedAtCompletion)),\"applied_seq\":\(detIdx),\"event\":\"apply\"}")
            } else {
                ignored += 1
            }
        }
        guard CVPixelBufferLockBaseAddress(imgBuf, .readOnly) == kCVReturnSuccess else {
            ioError("failed to lock frame \(k)")
        }
        guard let base = CVPixelBufferGetBaseAddress(imgBuf) else {
            CVPixelBufferUnlockBaseAddress(imgBuf, .readOnly)
            ioError("failed to access frame \(k)")
        }
        let bpr = CVPixelBufferGetBytesPerRow(imgBuf)
        let pyramid = LumaPyramid.fromBGRA(base, width: w, height: h, bytesPerRow: bpr)
        CVPixelBufferUnlockBaseAddress(imgBuf, .readOnly)
        tracker.ingestFrame(seq: UInt32(k), pts: ptsK, pyramid: pyramid)
        scheduler.frameIngested(k: k, pts: ptsK)
        let rendered = tracker.renderBoxes(now: ptsK, presentLead: presentLead)
        let appliedStr: String = lastOk.map { String($0) } ?? "null"
        emitLine("{\"frame\":\(k),\"pts\":\(ptsK),\"display_pts\":\(ptsK + presentLead),\"boxes\":\(boxesJSON(rendered)),\"applied_seq\":\(appliedStr)}")
        processed += 1
    }
    videoTotal += 1
}

if reader.status == .failed || reader.status == .cancelled {
    let msg = reader.error?.localizedDescription ?? "reader failed"
    ioError("failed to decode video: \(msg)")
}

if videoTotal != nUse {
    if abs(videoTotal - nUse) > 2 {
        ioError("video has \(videoTotal) frames but boxes has \(nUse); mismatch > 2")
    } else {
        warn("video has \(videoTotal) frames but boxes has \(nUse); using \(min(videoTotal, nUse))")
    }
}

let meanLatency = scheduler.meanLatency
emitLine("{\"type\":\"summary\",\"requests\":\(scheduler.requests),\"applied\":\(applied),\"ignored\":\(ignored),\"mean_latency_s\":\(meanLatency)}")
flushBuffer()
outHandle.closeFile()
