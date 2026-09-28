import Foundation

/// Offline replay of the live frame-credit pipeline (bench contract).
///
/// Clock = recording pts (real-time playback). One outstanding detection request.
/// - Start: request frame 0 when it is ingested.
/// - A request for frame i issued at time t completes at t + detectS[i] + overheadS.
/// - `beginFrame(k:pts:)` processes every pending completion with
///   completion time <= pts (in time order) and, after each completion, issues
///   the next request for the newest ingested frame at the completion time,
///   unless that frame was already requested (then sets creditPending).
/// - `frameIngested(k:pts:)` issues a request for k at pts when k == 0 or
///   creditPending is set.
///
/// Pure logic, no AVFoundation. Deterministic and unit-testable.
public struct ReplayScheduler {
    public let detectS: [Double]
    public let overheadS: Double

    public private(set) var requests: Int = 0
    public private(set) var completed: Int = 0
    /// Sum of (completion - pts of requested frame) over completed requests.
    public private(set) var latencySum: Double = 0
    public private(set) var creditPending: Bool = false

    public var meanLatency: Double {
        completed > 0 ? latencySum / Double(completed) : 0
    }

    private struct Pending {
        var frame: Int
        var issue: Double
        var completion: Double
        var framePts: Double
    }

    private var pending: Pending?
    private var requested: Set<Int> = []
    private var ptsByFrame: [Int: Double] = [:]
    private var newestIngested: Int?

    public init(detectS: [Double], overheadS: Double) {
        var ov = overheadS
        if ov.isNaN { ov = 0 }
        if ov < 0 { ov = 0 }
        self.detectS = detectS.map { d in
            if d.isNaN { return 0 }
            return d < 0 ? 0 : d
        }
        self.overheadS = ov
    }

    /// Record pts for k and return detection frame indices whose completion
    /// time <= pts, in completion order. Issues follow-up requests per contract.
    @discardableResult
    public mutating func beginFrame(k: Int, pts: Double) -> [Int] {
        beginFrameTimed(k: k, pts: pts).map { $0.frame }
    }

    /// Timed variant of `beginFrame`: returns (frame, completionTime) pairs
    /// in completion order so callers can re-render at the exact completion
    /// time (live parity). `beginFrame` delegates to this.
    @discardableResult
    public mutating func beginFrameTimed(k: Int, pts: Double) -> [(frame: Int, completion: Double)] {
        ptsByFrame[k] = pts
        var done: [(frame: Int, completion: Double)] = []
        while let p = pending, p.completion <= pts {
            pending = nil
            done.append((frame: p.frame, completion: p.completion))
            completed += 1
            latencySum += p.completion - p.framePts
            let newest = newestIngested
            if let ni = newest {
                if requested.contains(ni) {
                    creditPending = true
                } else {
                    issue(frame: ni, at: p.completion)
                }
            }
        }
        return done
    }

    /// Record ingestion of frame k; issues a request for k if this is frame 0
    /// or a credit is pending.
    public mutating func frameIngested(k: Int, pts: Double) {
        ptsByFrame[k] = pts
        if let ni = newestIngested {
            newestIngested = max(ni, k)
        } else {
            newestIngested = k
        }
        if pending != nil && !creditPending {
            // An outstanding request exists and no credit is owed: nothing to do.
            // (Frame 0 while a request is pending cannot happen in normal order.)
            return
        }
        if k == 0 {
            if !requested.contains(0) && pending == nil {
                creditPending = false
                issue(frame: 0, at: pts)
            }
            return
        }
        if creditPending {
            if !requested.contains(k) && pending == nil {
                creditPending = false
                issue(frame: k, at: pts)
            }
        }
    }

    private mutating func issue(frame: Int, at issueTime: Double) {
        let d: Double = (frame >= 0 && frame < detectS.count) ? detectS[frame] : 0
        let fpts = ptsByFrame[frame] ?? issueTime
        let comp = issueTime + d + overheadS
        pending = Pending(frame: frame, issue: issueTime, completion: comp, framePts: fpts)
        requested.insert(frame)
        requests += 1
    }
}
