import BsafeCore
import CoreImage
import CoreMedia
import Foundation
import ScreenCaptureKit

private struct PipelineStats {
    var frames: Int = 0
    var ingestMs: Double = 0
    var ageMs: Double = 0
    var sent: Int = 0
    var detApplied: Int = 0
    var detIgnored: Int = 0
    var catchupMs: Double = 0
    var catchupN: Int = 0
    var pixelMs: Double = 0
    var pixelN: Int = 0
}

/// Per-display live-v2 pipeline. All tracker/pipeline state lives on `queue`,
/// which is also the SCStream sampleHandlerQueue, so no locks are needed for
/// that state. Stats counters use a small lock for the utility-timer swap.
final class DisplayPipeline: NSObject, SCStreamOutput {
    let displayID: UInt32
    let queue: DispatchQueue
    let captureWidth: Int
    let captureHeight: Int
    let pointsWidth: Int
    let pointsHeight: Int
    let scaleFactor: Double
    let presentLead: Double
    let overlay: CensorOverlay
    let statsEnabled: Bool

    private let tracker: DisplayTracker
    private let ciContext: CIContext
    private let styleProvider: () -> CensorStyle
    private let frameSender: (UInt32, UInt32, UInt32, Int64, UInt32, Data) -> Void

    private let streamLock = NSLock()
    /// Current stream, guarded by streamLock (touched from the pipeline queue
    /// and the serial restart queue). Tracker/overlay/credit/seq state lives
    /// on `queue` and survives restarts; only the stream object is replaced.
    private var stream: SCStream?
    private var pipelineDelegate: PipelineStreamDelegate?
    private let restartQueue: DispatchQueue
    private var fpsForRestart: Int = 0
    private var excludedWindowsProvider: (() -> [SCWindow])?
    private var restartAttempt: Int = 0
    /// First stop of the current outage. Set once per outage and cleared only
    /// after >= 3 s of continuous frames on the new stream, so transient
    /// successes do not reset the 30 s total-outage cap.
    private var outageStart: CFAbsoluteTime?
    private var displayGoneSince: CFAbsoluteTime?
    private var failureTimes: [CFAbsoluteTime] = []
    /// First frame time on the current generation after a restart. The outage
    /// counters clear only once frames have flowed for >= 3 s.
    private var firstFrameAfterRestart: CFAbsoluteTime?
    /// Watchdog (not on `restartQueue`) enforcing the 30 s outage cap even
    /// when the restart queue is blocked. Guarded by streamLock.
    private var watchdog: DispatchSourceTimer?
    private var seq: UInt32 = 0
    private var latestBuffer: CVImageBuffer?
    private var latestSeq: UInt32?
    private var latestPts: Double?
    private var lastSentSeq: UInt32?
    private var creditPending = false
    private var recentPts: [(seq: UInt32, pts: Double)] = []

    private let statsLock = NSLock()
    private var stats = PipelineStats()

    init(
        displayID: UInt32,
        captureWidth: Int,
        captureHeight: Int,
        pointsWidth: Int,
        pointsHeight: Int,
        scaleFactor: Double,
        presentLead: Double,
        config: TrackerConfig,
        overlay: CensorOverlay,
        statsEnabled: Bool,
        styleProvider: @escaping () -> CensorStyle,
        frameSender: @escaping (UInt32, UInt32, UInt32, Int64, UInt32, Data) -> Void
    ) {
        self.displayID = displayID
        self.queue = DispatchQueue(label: "bsafe.display.\(displayID)")
        self.captureWidth = captureWidth
        self.captureHeight = captureHeight
        self.pointsWidth = pointsWidth
        self.pointsHeight = pointsHeight
        self.scaleFactor = scaleFactor
        self.presentLead = presentLead
        self.tracker = DisplayTracker(frameWidth: captureWidth, frameHeight: captureHeight, config: config)
        self.ciContext = CIContext()
        self.overlay = overlay
        self.statsEnabled = statsEnabled
        self.styleProvider = styleProvider
        self.frameSender = frameSender
        self.restartQueue = DispatchQueue(label: "bsafe.restart.\(displayID)")
        super.init()
        self.pipelineDelegate = PipelineStreamDelegate(pipeline: self)
    }

    /// Restart config for this display. Must be set before startStream so a
    /// later interruption can re-create the stream with the same parameters.
    func configureRestart(fps: Int, excludedWindowsProvider: @escaping () -> [SCWindow]) {
        streamLock.lock()
        fpsForRestart = fps
        self.excludedWindowsProvider = excludedWindowsProvider
        streamLock.unlock()
    }

    /// Create and synchronously start one SCStream for `display` using this
    /// pipeline's own delegate. Blocks until startCapture completes; throws
    /// on failure. The tracker/overlay/credit/seq state is untouched, so this
    /// is safe to call again after an interruption.
    func startStream(display: SCDisplay, fps: Int, excludingWindows: [SCWindow]) throws {
        guard let delegate = pipelineDelegate else {
            throw NSError(
                domain: "BsafeCapture", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Pipeline delegate missing"]
            )
        }
        let filter = SCContentFilter(display: display, excludingWindows: excludingWindows)
        let config = SCStreamConfiguration()
        config.width = captureWidth
        config.height = captureHeight
        config.captureResolution = .best
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.showsCursor = false
        config.queueDepth = 6
        config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(fps))

        let st = SCStream(filter: filter, configuration: config, delegate: delegate)
        try st.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)

        // Publish before startCapture so a stop arriving during start is not
        // dropped (it compares identical to the starting stream and retries).
        // Cleared again below if start fails.
        streamLock.lock()
        stream = st
        firstFrameAfterRestart = nil
        streamLock.unlock()

        let sem = DispatchSemaphore(value: 0)
        var captureError: Error?
        st.startCapture { err in
            captureError = err
            sem.signal()
        }
        if sem.wait(timeout: .now() + 5) == .timedOut {
            streamLock.lock()
            if stream === st {
                stream = nil
            }
            streamLock.unlock()
            throw NSError(
                domain: "BsafeCapture", code: 9,
                userInfo: [NSLocalizedDescriptionKey: "startCapture timed out"]
            )
        }
        if let captureError {
            streamLock.lock()
            if stream === st {
                stream = nil
            }
            streamLock.unlock()
            throw captureError
        }
        // A stop for this generation may have arrived during startCapture and
        // nilled the pointer; treat that as a failure so the stop's scheduled
        // retry (not this dead stream) wins.
        streamLock.lock()
        let stillCurrent = (stream === st)
        streamLock.unlock()
        if !stillCurrent {
            throw NSError(
                domain: "BsafeCapture", code: 10,
                userInfo: [NSLocalizedDescriptionKey: "stream stopped during start"]
            )
        }
    }

    func stopStream() {
        streamLock.lock()
        let st = stream
        streamLock.unlock()
        st?.stopCapture { _ in }
    }

    /// Entry point for stream interruptions (SCStreamDelegate callback and the
    /// SIGUSR1 test hook share this path). Ignores stops for any stream that
    /// is not the current generation, keeps the original `outageStart` across
    /// transient successes, counts every genuine stop as a failure, and gives
    /// up with _exit(3) past 30 s total outage or >20 failures in 120 s. Logs
    /// one line per stop. Callable from any thread; never blocks the main
    /// thread (uses _exit, never exit, from background queues).
    func handleStreamStopped(code: Int, stream stoppedStream: SCStream? = nil) {
        let now = CFAbsoluteTimeGetCurrent()
        streamLock.lock()
        let current = stream
        if let stoppedStream, let current, stoppedStream !== current {
            streamLock.unlock()
            return // stale stop for a previous generation
        }
        if current == nil, outageStart != nil {
            streamLock.unlock()
            return // already restarting; retry already scheduled/in flight
        }
        let isFirst = (outageStart == nil)
        if isFirst {
            outageStart = now
            restartAttempt = 0
            displayGoneSince = nil
        }
        // A startCapture success followed by a stop counts as a failure.
        failureTimes.append(now)
        failureTimes = failureTimes.filter { now - $0 <= 120 }
        let outageFor = now - (outageStart ?? now)
        let failCount = failureTimes.filter { now - $0 <= 120 }.count
        stream = nil
        firstFrameAfterRestart = nil
        let needWatchdog = (watchdog == nil)
        streamLock.unlock()
        if needWatchdog {
            startWatchdog()
        }
        if outageFor > 30 || failCount > 20 {
            fputs("BsafeCapture: outage \(String(format: "%.1f", outageFor))s / \(failCount) failures in 120s, giving up\n", stderr)
            _exit(3)
        }
        fputs("BsafeCapture: stream stopped (code \(code)), restarting…\n", stderr)
        scheduleNextRestart()
    }

    /// Debug hook: stop the stream and run the same restart path as
    /// didStopWithError with synthetic code -1 (stopCapture alone does not
    /// call didStopWithError).
    func simulateInterruption() {
        streamLock.lock()
        let st = stream
        streamLock.unlock()
        st?.stopCapture { _ in }
        handleStreamStopped(code: -1, stream: st)
    }

    /// Watchdog timer (not on `restartQueue`) enforcing the 30 s total-outage
    /// cap even when the restart queue is blocked (e.g. a stalled
    /// SCShareableContent callback). Fires every second until the outage
    /// clears after >= 3 s of frames.
    private func startWatchdog() {
        streamLock.lock()
        if watchdog != nil {
            streamLock.unlock()
            return
        }
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + 1.0, repeating: 1.0)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            self.streamLock.lock()
            guard let since = self.outageStart else {
                self.streamLock.unlock()
                return
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - since
            self.streamLock.unlock()
            if elapsed > 30 {
                fputs("BsafeCapture: outage >30s without recovery, giving up\n", stderr)
                _exit(3)
            }
        }
        watchdog = timer
        streamLock.unlock()
        timer.resume()
    }

    private func scheduleNextRestart() {
        streamLock.lock()
        let next = restartAttempt + 1
        streamLock.unlock()
        let delay = restartDelay(forAttempt: next)
        restartQueue.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.attemptRestart(attempt: next)
        }
    }

    private func attemptRestart(attempt: Int) {
        streamLock.lock()
        let current = stream
        let since = outageStart
        streamLock.unlock()
        if current != nil || since == nil {
            return // already back, or no outage
        }
        let now = CFAbsoluteTimeGetCurrent()
        // Total-outage cap measured from the FIRST stop, never reset by
        // transient successes (only the 3 s healthy-frames path clears it).
        if now - (since ?? now) > 30 {
            fputs("BsafeCapture: outage >30s without recovery, giving up\n", stderr)
            _exit(3)
        }

        guard let fresh = fetchDisplay(displayID: displayID) else {
            recordFailure(at: now)
            streamLock.lock()
            if displayGoneSince == nil {
                displayGoneSince = now
            }
            let goneFor = now - (displayGoneSince ?? now)
            let outageFor = now - (outageStart ?? now)
            streamLock.unlock()
            if goneFor > 30 || outageFor > 30 || recentFailureCount(now: now) > 20 {
                fputs("BsafeCapture: display \(displayID) gone for >30s or restart failed 20+ times in 120s, giving up\n", stderr)
                _exit(3)
            }
            streamLock.lock()
            restartAttempt = attempt
            streamLock.unlock()
            scheduleNextRestart()
            return
        }
        streamLock.lock()
        displayGoneSince = nil
        let restartFps = fpsForRestart
        let provider = excludedWindowsProvider
        streamLock.unlock()

        let expectedW = Int((Double(fresh.width) * scaleFactor).rounded())
        let expectedH = Int((Double(fresh.height) * scaleFactor).rounded())
        if expectedW != captureWidth || expectedH != captureHeight {
            fputs(
                "BsafeCapture: display \(displayID) size changed (\(expectedW)x\(expectedH) vs \(captureWidth)x\(captureHeight)), giving up\n",
                stderr
            )
            _exit(3)
        }

        let excluded = provider?() ?? []
        do {
            try startStream(display: fresh, fps: restartFps, excludingWindows: excluded)
        } catch let err as NSError where err.domain == "BsafeCapture" && err.code == 10 {
            // Stop arrived during startCapture: handleStreamStopped already
            // recorded the failure and scheduled the retry.
            return
        } catch {
            recordFailure(at: now)
            streamLock.lock()
            let outageFor = CFAbsoluteTimeGetCurrent() - (outageStart ?? now)
            streamLock.unlock()
            if outageFor > 30 || recentFailureCount(now: now) > 20 {
                fputs("BsafeCapture: stream restart failed 20+ times in 120s or outage >30s, giving up\n", stderr)
                _exit(3)
            }
            streamLock.lock()
            restartAttempt = attempt
            streamLock.unlock()
            scheduleNextRestart()
            return
        }
        // A stop may have landed during startCapture and nilled the pointer
        // (startStream throws in that case, but re-check for the race where
        // the stop ran just after it returned).
        streamLock.lock()
        let stillCurrent = (stream != nil)
        streamLock.unlock()
        if !stillCurrent {
            return // the stop already scheduled the next retry
        }
        // Same pixel size with a new global origin (arrangement change):
        // move the overlay on the main thread instead of exiting.
        overlay.refreshFrame(displayID: displayID)
        let elapsed = CFAbsoluteTimeGetCurrent() - (since ?? now)
        fputs(String(format: "BsafeCapture: stream restarted after %.1fs (attempt %d)\n", elapsed, attempt), stderr)
        streamLock.lock()
        // Kept until >= 3 s of continuous frames clears the outage; a stop
        // before then counts as another failure without resetting outageStart.
        restartAttempt = attempt
        streamLock.unlock()
    }

    private func recordFailure(at t: CFAbsoluteTime) {
        streamLock.lock()
        failureTimes.append(t)
        failureTimes = failureTimes.filter { t - $0 <= 120 }
        streamLock.unlock()
    }

    private func recentFailureCount(now: CFAbsoluteTime) -> Int {
        streamLock.lock()
        defer { streamLock.unlock() }
        return failureTimes.filter { now - $0 <= 120 }.count
    }

    // MARK: - SCStreamOutput (runs on `queue`)

    func stream(
        _ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .screen else { return }
        // Ignore stale buffers from a previous stream generation: only the
        // current stream may feed the tracker (the pts jump on the first new
        // frame after an outage is fine — the tracker is time-based).
        // NOTE: `stream` here is the callback parameter; self.stream is the
        // current stream under the lock.
        streamLock.lock()
        let currentStream = self.stream
        let inOutage = outageStart != nil
        streamLock.unlock()
        guard let currentStream, stream === currentStream else { return }
        // Accept only complete frames.
        var complete = false
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as NSArray?,
            attachments.count > 0,
            let dict = attachments[0] as? NSDictionary
        {
            let wantKey = SCStreamFrameInfo.status.rawValue
            for (k, v) in dict {
                if let ks = k as? String, ks == wantKey {
                    if let num = v as? Int, num == SCFrameStatus.complete.rawValue {
                        complete = true
                    }
                    break
                }
            }
        }
        guard complete else { return }
        guard let imageBuffer = sampleBuffer.imageBuffer else { return }
        if inOutage {
            // The outage is over only after >= 3 s of continuous complete
            // frames on the new stream. A startCapture success followed by a
            // quick stop (or an occasionally-emitting stream) must not clear
            // the failure counters or the 30 s total-outage clock.
            let now = CFAbsoluteTimeGetCurrent()
            streamLock.lock()
            if firstFrameAfterRestart == nil {
                firstFrameAfterRestart = now
            }
            let healthyFor = now - (firstFrameAfterRestart ?? now)
            let healthy = healthyFor >= 3.0
            if healthy {
                outageStart = nil
                restartAttempt = 0
                failureTimes = []
                displayGoneSince = nil
                firstFrameAfterRestart = nil
            }
            let timer = healthy ? watchdog : nil
            if healthy {
                watchdog = nil
            }
            streamLock.unlock()
            timer?.cancel()
        }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let hostNow = CMClockGetTime(CMClockGetHostTimeClock()).seconds
        let w = CVPixelBufferGetWidth(imageBuffer)
        let h = CVPixelBufferGetHeight(imageBuffer)

        guard CVPixelBufferLockBaseAddress(imageBuffer, .readOnly) == kCVReturnSuccess else { return }
        guard let base = CVPixelBufferGetBaseAddress(imageBuffer) else {
            CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)
            return
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)

        let t0 = CFAbsoluteTimeGetCurrent()
        let pyramid = LumaPyramid.fromBGRA(base, width: w, height: h, bytesPerRow: bytesPerRow)
        seq &+= 1
        let mySeq = seq
        tracker.ingestFrame(seq: mySeq, pts: pts, pyramid: pyramid)
        let ingestMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

        var sendCopy: Data?
        if creditPending {
            sendCopy = copyPacked(base: base, width: w, height: h, bytesPerRow: bytesPerRow)
            creditPending = false
            lastSentSeq = mySeq
        }
        CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)

        latestBuffer = imageBuffer
        latestSeq = mySeq
        latestPts = pts
        recentPts.append((seq: mySeq, pts: pts))
        if recentPts.count > 96 {
            recentPts.removeFirst(recentPts.count - 96)
        }

        if statsEnabled {
            statsLock.lock()
            stats.frames += 1
            stats.ingestMs += ingestMs
            stats.ageMs += (hostNow - pts) * 1000.0
            statsLock.unlock()
        }

        if let pixels = sendCopy {
            let ptsNs = Int64((pts * 1_000_000_000.0).rounded())
            frameSender(displayID, UInt32(w), UInt32(h), ptsNs, mySeq, pixels)
            if statsEnabled {
                statsLock.lock()
                stats.sent += 1
                statsLock.unlock()
            }
        }

        publishRender()
    }

    /// Granted one frame credit (called via queue.async from the reader thread).
    func handleCredit() {
        if let buf = latestBuffer, let ls = latestSeq, ls != lastSentSeq {
            let w = CVPixelBufferGetWidth(buf)
            let h = CVPixelBufferGetHeight(buf)
            guard CVPixelBufferLockBaseAddress(buf, .readOnly) == kCVReturnSuccess else {
                creditPending = true
                return
            }
            guard let base = CVPixelBufferGetBaseAddress(buf) else {
                CVPixelBufferUnlockBaseAddress(buf, .readOnly)
                creditPending = true
                return
            }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buf)
            let pixels = copyPacked(base: base, width: w, height: h, bytesPerRow: bytesPerRow)
            CVPixelBufferUnlockBaseAddress(buf, .readOnly)
            lastSentSeq = ls
            if let pts = latestPts {
                let ptsNs = Int64((pts * 1_000_000_000.0).rounded())
                frameSender(displayID, UInt32(w), UInt32(h), ptsNs, ls, pixels)
                if statsEnabled {
                    statsLock.lock()
                    stats.sent += 1
                    statsLock.unlock()
                }
            }
        } else {
            creditPending = true
        }
    }

    /// Apply Python detections for frame `seq` (called via queue.async from the reader thread).
    func handleDetections(frameWidth: Int, frameHeight: Int, seq detSeq: UInt32, boxes: [TrackBox]) {
        guard frameWidth == captureWidth, frameHeight == captureHeight else {
            if statsEnabled {
                statsLock.lock()
                stats.detIgnored += 1
                statsLock.unlock()
            }
            return
        }
        let ok = tracker.applyDetections(seq: detSeq, boxes: boxes)
        if statsEnabled {
            statsLock.lock()
            if ok {
                stats.detApplied += 1
                if let lp = latestPts, let dp = ptsForSeq(detSeq) {
                    stats.catchupMs += (lp - dp) * 1000.0
                    stats.catchupN += 1
                }
            } else {
                stats.detIgnored += 1
            }
            statsLock.unlock()
        }
        publishRender()
    }

    /// Build the MSG_STATS line and reset counters. Called on the utility timer.
    func takeStatsLine() -> String? {
        guard statsEnabled else { return nil }
        statsLock.lock()
        let s = stats
        stats = PipelineStats()
        statsLock.unlock()
        let drains = overlay.takeDrainCount()
        let interval = 2.0
        func avg(_ sum: Double, _ n: Int) -> Double {
            return n > 0 ? sum / Double(n) : 0.0
        }
        let line = String(
            format: "display=%u fps=%.1f ingest_ms=%.2f age_ms=%.1f sent=%d det=%d/%d catchup_ms=%.1f render_hz=%.1f pixelate_ms=%.2f",
            displayID,
            Double(s.frames) / interval,
            avg(s.ingestMs, s.frames),
            avg(s.ageMs, s.frames),
            s.sent,
            s.detApplied,
            s.detIgnored,
            avg(s.catchupMs, s.catchupN),
            Double(drains) / interval,
            avg(s.pixelMs, s.pixelN)
        )
        return line
    }

    // MARK: - Private (all on `queue` unless noted)

    private func ptsForSeq(_ target: UInt32) -> Double? {
        for entry in recentPts.reversed() where entry.seq == target {
            return entry.pts
        }
        return nil
    }

    private func copyPacked(base: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) -> Data {
        let rowBytes = width * 4
        var out = Data(count: rowBytes * height)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let d = dst.baseAddress else { return }
            for row in 0..<height {
                let s = base.advanced(by: row * bytesPerRow)
                let dd = d.advanced(by: row * rowBytes)
                memcpy(dd, s, rowBytes)
            }
        }
        return out
    }

    private func publishRender() {
        let style = styleProvider()
        let now = CMClockGetTime(CMClockGetHostTimeClock()).seconds
        let boxes = tracker.renderBoxes(now: now, presentLead: presentLead)

        var rects: [CGRect] = []
        rects.reserveCapacity(boxes.count)
        for b in boxes {
            let x0 = floor(b.x)
            let y0 = floor(b.y)
            let x1 = ceil(b.x + b.w)
            let y1 = ceil(b.y + b.h)
            rects.append(CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
        }

        var images: [CGImage?] = Array(repeating: nil, count: rects.count)
        if style.pixels > 0.0, let buf = latestBuffer, !rects.isEmpty {
            let t0 = CFAbsoluteTimeGetCurrent()
            let cw = captureWidth
            let ch = captureHeight
            let blockSize = max(2.0, style.pixels * 15.0 * scaleFactor)
            if CVPixelBufferLockBaseAddress(buf, .readOnly) == kCVReturnSuccess {
                let full = CIImage(cvImageBuffer: buf)
                for i in 0..<rects.count {
                    var ix = Int(rects[i].origin.x)
                    var iy = Int(rects[i].origin.y)
                    var iw = Int(rects[i].width)
                    var ih = Int(rects[i].height)
                    if ix < 0 { iw += ix; ix = 0 }
                    if iy < 0 { ih += iy; iy = 0 }
                    if ix + iw > cw { iw = cw - ix }
                    if iy + ih > ch { ih = ch - iy }
                    guard iw > 0, ih > 0 else {
                        images[i] = nil
                        continue
                    }
                    // CoreImage origin is bottom-left.
                    let crop = CGRect(x: ix, y: ch - iy - ih, width: iw, height: ih)
                    // Clamp so edge blocks sample edge pixels instead of transparent (black on the layer).
                    let cropped = full.cropped(to: crop).clampedToExtent()
                    guard let filt = CIFilter(name: "CIPixellate") else {
                        images[i] = nil
                        continue
                    }
                    filt.setValue(cropped, forKey: kCIInputImageKey)
                    filt.setValue(blockSize, forKey: kCIInputScaleKey)
                    filt.setValue(CIVector(x: 0, y: 0), forKey: kCIInputCenterKey)
                    if let out = filt.outputImage, let cg = ciContext.createCGImage(out, from: crop) {
                        images[i] = cg
                    } else {
                        images[i] = nil
                    }
                }
                CVPixelBufferUnlockBaseAddress(buf, .readOnly)
            }
            if statsEnabled {
                statsLock.lock()
                stats.pixelMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                stats.pixelN += 1
                statsLock.unlock()
            }
        }

        overlay.publish(OverlayFrame(
            boxes: rects,
            images: images,
            captureWidth: captureWidth,
            captureHeight: captureHeight,
            blur: style.blur,
            pixels: style.pixels,
            text: style.text
        ))
    }
}
