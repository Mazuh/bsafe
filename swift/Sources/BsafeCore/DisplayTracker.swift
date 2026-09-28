import Foundation

public struct TrackerConfig {
    public var persistPasses: Int = 8
    /// Per-edge blend for matched detections: an edge the detection moves outward snaps to it;
    /// an edge it moves inward moves by shrinkAlpha of the difference per 0.1 s of detection time.
    public var shrinkAlpha: Double = 0.1
    /// Minimum detection-time a track persists after its last match, even if miss count is exceeded.
    public var minPersistSeconds: Double = 1.0
    public var historyCapacity: Int = 48
    public var frameSearchRadius: Int = 12
    public var catchUpSearchRadius: Int = 24
    public var maxLeadSeconds: Double = 0.1

    public init(persistPasses: Int = 8, shrinkAlpha: Double = 0.1, minPersistSeconds: Double = 1.0) {
        self.persistPasses = max(1, persistPasses)
        var a = shrinkAlpha
        if a.isNaN { a = 0.1 }
        if a < 0 { a = 0 }
        if a > 1 { a = 1 }
        self.shrinkAlpha = a
        var m = minPersistSeconds
        if m.isNaN { m = 1.0 }
        if m < 0 { m = 0 }
        self.minPersistSeconds = m
        self.historyCapacity = 48
        self.frameSearchRadius = 12
        self.catchUpSearchRadius = 24
        self.maxLeadSeconds = 0.1
    }
}

public struct Track {
    public let id: Int
    public var box: TrackBox
    public var velocity: Shift
    public var misses: Int
    var coastCount: Int = 0
    /// Tight box used for motion estimation: the unblended primed detection, moved with content.
    var core: TrackBox
    /// Pts of the history entry that created or last matched this track.
    var lastMatchPts: Double = 0

    public init(id: Int, box: TrackBox, velocity: Shift, misses: Int = 0) {
        self.id = id
        self.box = box
        self.velocity = velocity
        self.misses = misses
        self.coastCount = 0
        self.core = box
        self.lastMatchPts = 0
    }
}

private struct HistoryEntry {
    var seq: UInt32
    var pts: Double
    var pyr: LumaPyramid
}

public final class DisplayTracker {
    public let frameWidth: Int
    public let frameHeight: Int
    public private(set) var config: TrackerConfig
    public private(set) var tracks: [Track]
    public private(set) var latestSeq: UInt32?
    public private(set) var latestPts: Double?

    private var history: [HistoryEntry]
    private var nextId: Int

    public init(frameWidth: Int, frameHeight: Int, config: TrackerConfig = TrackerConfig()) {
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.config = config
        self.tracks = []
        self.history = []
        self.nextId = 0
        self.latestSeq = nil
        self.latestPts = nil
    }

    public func ingestFrame(seq: UInt32, pts: Double, pyramid: LumaPyramid) {
        if let last = history.last {
            if last.pyr.captureWidth != pyramid.captureWidth
                || last.pyr.captureHeight != pyramid.captureHeight
            {
                history.removeAll()
                history.append(HistoryEntry(seq: seq, pts: pts, pyr: pyramid))
                latestSeq = seq
                latestPts = pts
                return
            }
            var dt = pts - last.pts
            if !(dt > 0) { dt = 1.0 / 60.0 } else if dt > 0.1 { dt = 0.1 }

            let n = tracks.count
            var results = [Shift?](repeating: nil, count: n)
            var measuredShifts = [Shift]()
            var measuredVels = [Shift]()
            measuredShifts.reserveCapacity(n)
            measuredVels.reserveCapacity(n)
            for i in 0..<n {
                let tr = tracks[i]
                let pred = Shift(dx: tr.velocity.dx * dt, dy: tr.velocity.dy * dt)
                if let s = MotionEstimator.estimate(
                    from: last.pyr, to: pyramid, box: DisplayTracker.estimationRegion(tr.core, frameWidth: frameWidth, frameHeight: frameHeight),
                    predicted: pred, searchRadius: config.frameSearchRadius
                ) {
                    results[i] = s
                    measuredShifts.append(s)
                    let nv: Shift
                    if abs(s.dx) < 0.5 && abs(s.dy) < 0.5 {
                        nv = Shift.zero
                    } else {
                        nv = Shift(
                            dx: 0.7 * (s.dx / dt) + 0.3 * tr.velocity.dx,
                            dy: 0.7 * (s.dy / dt) + 0.3 * tr.velocity.dy
                        )
                    }
                    measuredVels.append(nv)
                }
            }
            var medShift: Shift? = nil
            var medVel: Shift? = nil
            if !measuredShifts.isEmpty {
                medShift = Shift(
                    dx: DisplayTracker.median(measuredShifts.map { $0.dx }) ?? 0,
                    dy: DisplayTracker.median(measuredShifts.map { $0.dy }) ?? 0
                )
                medVel = Shift(
                    dx: DisplayTracker.median(measuredVels.map { $0.dx }) ?? 0,
                    dy: DisplayTracker.median(measuredVels.map { $0.dy }) ?? 0
                )
            }
            for i in 0..<n {
                if let s = results[i] {
                    tracks[i].box = tracks[i].box.offsetBy(dx: s.dx, dy: s.dy)
                    tracks[i].core = tracks[i].core.offsetBy(dx: s.dx, dy: s.dy)
                    if abs(s.dx) < 0.5 && abs(s.dy) < 0.5 {
                        tracks[i].velocity = Shift.zero
                    } else {
                        let ov = tracks[i].velocity
                        // ov is still old value (not yet updated for this i)
                        tracks[i].velocity = Shift(
                            dx: 0.7 * (s.dx / dt) + 0.3 * ov.dx,
                            dy: 0.7 * (s.dy / dt) + 0.3 * ov.dy
                        )
                    }
                    tracks[i].coastCount = 0
                } else {
                    if let ms = medShift, let mv = medVel {
                        tracks[i].box = tracks[i].box.offsetBy(dx: ms.dx, dy: ms.dy)
                        tracks[i].core = tracks[i].core.offsetBy(dx: ms.dx, dy: ms.dy)
                        tracks[i].velocity = mv
                        tracks[i].coastCount = 0
                    } else {
                        let v = tracks[i].velocity
                        if v == Shift.zero {
                            tracks[i].coastCount = 0
                        } else if tracks[i].coastCount >= 45 {
                            tracks[i].velocity = Shift.zero
                        } else {
                            let b = tracks[i].box
                            let basePad = max(8.0, 0.25 * max(b.w, b.h))
                            let padX = basePad + abs(v.dx * dt)
                            let padY = basePad + abs(v.dy * dt)
                            let gx0 = b.x - padX
                            let gy0 = b.y - padY
                            let gx1 = b.x + b.w + padX
                            let gy1 = b.y + b.h + padY
                            let fw = Double(frameWidth)
                            let fh = Double(frameHeight)
                            if gx0 <= 0 || gy0 <= 0 || gx1 >= fw || gy1 >= fh {
                                tracks[i].box = b.offsetBy(dx: v.dx * dt, dy: v.dy * dt)
                                tracks[i].core = tracks[i].core.offsetBy(dx: v.dx * dt, dy: v.dy * dt)
                                tracks[i].coastCount += 1
                            } else {
                                tracks[i].velocity = Shift.zero
                                tracks[i].coastCount = 0
                            }
                        }
                    }
                }
            }
            tracks.removeAll { $0.box.clipped(width: Double(frameWidth), height: Double(frameHeight)) == nil }
            history.append(HistoryEntry(seq: seq, pts: pts, pyr: pyramid))
            if history.count > config.historyCapacity {
                history.removeFirst(history.count - config.historyCapacity)
            }
            latestSeq = seq
            latestPts = pts
        } else {
            history.append(HistoryEntry(seq: seq, pts: pts, pyr: pyramid))
            latestSeq = seq
            latestPts = pts
        }
    }

    @discardableResult
    public func applyDetections(seq: UInt32, boxes: [TrackBox]) -> Bool {
        guard let idx = history.lastIndex(where: { $0.seq == seq }) else { return false }
        guard let latest = history.last,
              let latestSeqV = latestSeq,
              let latestPtsV = latestPts
        else { return false }
        let entry = history[idx]
        let gap = latestPtsV - entry.pts
        let isNewest = (idx == history.count - 1)

        let medExisting: Shift? = tracks.isEmpty ? nil : Shift(
            dx: DisplayTracker.median(tracks.map { $0.velocity.dx }) ?? 0,
            dy: DisplayTracker.median(tracks.map { $0.velocity.dy }) ?? 0
        )

        // Catch-up shifts
        var shifts = [Shift](repeating: Shift.zero, count: boxes.count)
        if !boxes.isEmpty {
            if isNewest {
                for i in 0..<boxes.count { shifts[i] = Shift.zero }
            } else {
                let predictedBase: Shift
                if let mv = medExisting {
                    predictedBase = Shift(dx: mv.dx * gap, dy: mv.dy * gap)
                } else {
                    predictedBase = Shift.zero
                }
                var first = [Shift?](repeating: nil, count: boxes.count)
                var successes = [Shift]()
                for i in 0..<boxes.count {
                    if let s = MotionEstimator.estimate(
                        from: entry.pyr, to: latest.pyr,
                        box: DisplayTracker.estimationRegion(boxes[i], frameWidth: frameWidth, frameHeight: frameHeight),
                        predicted: predictedBase, searchRadius: config.catchUpSearchRadius
                    ) {
                        first[i] = s
                        successes.append(s)
                    }
                }
                let medSuccess: Shift? = successes.isEmpty ? nil : Shift(
                    dx: DisplayTracker.median(successes.map { $0.dx }) ?? 0,
                    dy: DisplayTracker.median(successes.map { $0.dy }) ?? 0
                )
                for i in 0..<boxes.count {
                    if let s = first[i] { shifts[i] = s }
                    else if let ms = medSuccess { shifts[i] = ms }
                    else { shifts[i] = predictedBase }
                }
            }
        }

        var primed = [TrackBox]()
        primed.reserveCapacity(boxes.count)
        for i in 0..<boxes.count {
            primed.append(boxes[i].offsetBy(dx: shifts[i].dx, dy: shifts[i].dy))
        }

        // Greedy matching
        struct Pair {
            var ti: Int
            var di: Int
            var score: Double
        }
        var pairs = [Pair]()
        for ti in 0..<tracks.count {
            for di in 0..<primed.count {
                let io = tracks[ti].box.iou(primed[di])
                if io >= 0.1 {
                    pairs.append(Pair(ti: ti, di: di, score: io))
                } else {
                    let tcx = tracks[ti].box.centerX
                    let tcy = tracks[ti].box.centerY
                    let dcx = primed[di].centerX
                    let dcy = primed[di].centerY
                    if tracks[ti].box.contains(x: dcx, y: dcy)
                        || primed[di].contains(x: tcx, y: tcy)
                    {
                        pairs.append(Pair(ti: ti, di: di, score: 0.05))
                    }
                }
            }
        }
        pairs.sort { $0.score > $1.score }
        var matchedT = Set<Int>()
        var matchedD = Set<Int>()
        var matchForT = [Int: Int]() // ti -> di
        for p in pairs {
            if matchedT.contains(p.ti) || matchedD.contains(p.di) { continue }
            matchedT.insert(p.ti)
            matchedD.insert(p.di)
            matchForT[p.ti] = p.di
        }

        let k = config.shrinkAlpha
        // Update matched
        for ti in 0..<tracks.count {
            if let di = matchForT[ti] {
                let b2 = primed[di]
                let old = tracks[ti].box
                // Capped at one reference step so a match after a gap (or a pts jump) cannot collapse the box at once;
                // a late detection for an older frame never shrinks it.
                let dt = min(entry.pts - tracks[ti].lastMatchPts, DisplayTracker.shrinkReferenceSeconds)
                let kEff: Double
                if dt > 0 {
                    var eff = 1.0 - pow(1.0 - k, dt / DisplayTracker.shrinkReferenceSeconds)
                    if eff < 0 { eff = 0 }
                    if eff > 1 { eff = 1 }
                    kEff = eff
                } else if dt == 0 {
                    kEff = k
                } else {
                    kEff = 0
                }
                tracks[ti].core = primed[di]
                tracks[ti].box = DisplayTracker.blendGrowShrink(old: old, det: b2, shrinkAlpha: kEff)
                tracks[ti].misses = 0
                tracks[ti].coastCount = 0
                tracks[ti].lastMatchPts = max(tracks[ti].lastMatchPts, entry.pts)
                let s = shifts[di]
                if tracks[ti].velocity == Shift.zero && gap > 0.001 && s != Shift.zero {
                    tracks[ti].velocity = Shift(dx: s.dx / gap, dy: s.dy / gap)
                }
            }
        }
        // New tracks for unmatched detections
        for di in 0..<primed.count {
            if matchedD.contains(di) { continue }
            let s = shifts[di]
            let vel: Shift
            if gap > 0.001 {
                vel = Shift(dx: s.dx / gap, dy: s.dy / gap)
            } else {
                vel = medExisting ?? Shift.zero
            }
            var t = Track(id: nextId, box: primed[di], velocity: vel, misses: 0)
            t.lastMatchPts = entry.pts
            nextId += 1
            tracks.append(t)
            // newly appended tracks are never in matchedT; no misses handling needed
        }
        // Unmatched tracks miss
        // Note: indices of original tracks are 0..<oldCount; new tracks appended after.
        // We need old count to avoid touching new tracks.
        // matchedT only refers to old indices, so iterate old range.
        // To get old count: total - (unmatched detection count). Recompute:
        let newCount = primed.count - matchedD.count
        let oldCount = tracks.count - newCount
        for ti in 0..<oldCount {
            if matchForT[ti] == nil {
                tracks[ti].misses += 1
            }
        }
        // Remove stale
        // Written as !(elapsed < min) so a NaN pts evicts instead of pinning the box.
        tracks.removeAll {
            $0.misses >= config.persistPasses
                && !((entry.pts - $0.lastMatchPts) < config.minPersistSeconds)
        }
        tracks.removeAll { $0.box.clipped(width: Double(frameWidth), height: Double(frameHeight)) == nil }
        _ = latestSeqV
        return true
    }

    public func renderBoxes(now: Double, presentLead: Double) -> [TrackBox] {
        guard let latestPtsV = latestPts else { return [] }
        var lead = (now - latestPtsV) + presentLead
        if lead < 0 { lead = 0 }
        if lead > config.maxLeadSeconds { lead = config.maxLeadSeconds }
        var out = [TrackBox]()
        out.reserveCapacity(tracks.count)
        let fw = Double(frameWidth)
        let fh = Double(frameHeight)
        for tr in tracks {
            let limit = min(0.75 * max(tr.box.w, tr.box.h), 400.0)
            var px = tr.velocity.dx * lead
            var py = tr.velocity.dy * lead
            if px < -limit { px = -limit }
            if px > limit { px = limit }
            if py < -limit { py = -limit }
            if py > limit { py = limit }
            let moved = tr.box.offsetBy(dx: px, dy: py)
            let u = tr.box.union(moved)
            if let c = u.clipped(width: fw, height: fh) {
                out.append(c)
            }
        }
        return out
    }

    public func reset() {
        tracks.removeAll()
        history.removeAll()
        latestSeq = nil
        latestPts = nil
    }

    /// Region motion is estimated on. Live boxes carry padding on every side (0.4 of the size per side by
    /// default, so the object is the central ~55%); sampling only the center keeps a static margin from
    /// out-voting a moving object. A core that crosses a frame edge uses the full core, so edge coasting is unchanged.
    static let estimationFraction = 0.55
    static let shrinkReferenceSeconds = 0.1
    static func estimationRegion(_ c: TrackBox, frameWidth: Int, frameHeight: Int) -> TrackBox {
        let f = estimationFraction
        let r = TrackBox(x: c.x + c.w * (1 - f) / 2, y: c.y + c.h * (1 - f) / 2, w: c.w * f, h: c.h * f)
        let inside = c.x >= 0 && c.y >= 0 && c.x + c.w <= Double(frameWidth) && c.y + c.h <= Double(frameHeight)
        return inside ? r : c
    }

    static func blendGrowShrink(old: TrackBox, det: TrackBox, shrinkAlpha: Double) -> TrackBox {
        let k = shrinkAlpha
        let left = det.x < old.x ? det.x : k * det.x + (1 - k) * old.x
        let top = det.y < old.y ? det.y : k * det.y + (1 - k) * old.y
        let oldRight = old.x + old.w
        let detRight = det.x + det.w
        let right = detRight > oldRight ? detRight : k * detRight + (1 - k) * oldRight
        let oldBottom = old.y + old.h
        let detBottom = det.y + det.h
        let bottom = detBottom > oldBottom ? detBottom : k * detBottom + (1 - k) * oldBottom
        return TrackBox(x: left, y: top, w: right - left, h: bottom - top)
    }

    static func median(_ vals: [Double]) -> Double? {
        if vals.isEmpty { return nil }
        let s = vals.sorted()
        let n = s.count
        if n % 2 == 1 {
            return s[n / 2]
        } else {
            return (s[n / 2 - 1] + s[n / 2]) / 2.0
        }
    }
}
