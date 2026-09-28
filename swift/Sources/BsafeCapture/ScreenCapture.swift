import Foundation
import ScreenCaptureKit

/// Per-pipeline SCStream delegate: forwards the stop to its pipeline so only
/// that display restarts. Never exits; the pipeline owns recovery.
final class PipelineStreamDelegate: NSObject, SCStreamDelegate {
    weak var pipeline: DisplayPipeline?

    init(pipeline: DisplayPipeline? = nil) {
        self.pipeline = pipeline
        super.init()
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        let code = (error as NSError).code
        pipeline?.handleStreamStopped(code: code, stream: stream)
    }
}

/// Synchronously resolve the displays selected by the --display filter.
/// Same selection semantics and errors as the original implementation.
/// Returns the selected displays plus the full shareable content (for app exclusion).
func resolveDisplays(filter rawFilter: String?) throws -> (displays: [SCDisplay], content: SCShareableContent) {
    var fetched: SCShareableContent?
    var fetchError: Error?
    let sem = DispatchSemaphore(value: 0)
    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, error in
        fetched = content
        fetchError = error
        sem.signal()
    }
    if sem.wait(timeout: .now() + 5) == .timedOut {
        throw NSError(
            domain: "BsafeCapture", code: 6,
            userInfo: [NSLocalizedDescriptionKey: "Timed out fetching shareable content"]
        )
    }
    if let fetchError {
        throw fetchError
    }
    guard let content = fetched, !content.displays.isEmpty else {
        throw NSError(
            domain: "BsafeCapture", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "No displays found"]
        )
    }

    let primaryID = CGMainDisplayID()
    let filter = rawFilter ?? "primary"
    let selected: [SCDisplay]

    switch filter {
    case "primary":
        if let primary = content.displays.first(where: { $0.displayID == primaryID }) {
            selected = [primary]
        } else {
            // No primary found; fall back to first available display
            selected = [content.displays[0]]  // safe: displays.isEmpty checked above
        }
    case "secondary":
        let nonPrimary = content.displays.filter { $0.displayID != primaryID }
        guard !nonPrimary.isEmpty else {
            throw NSError(
                domain: "BsafeCapture", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "No secondary display found"]
            )
        }
        selected = [nonPrimary[0]]
    case "all":
        selected = content.displays
    default:
        // Numeric display ID
        if let numericID = UInt32(filter) {
            guard let match = content.displays.first(where: { $0.displayID == numericID }) else {
                throw NSError(
                    domain: "BsafeCapture", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "Display with ID \(filter) not found"]
                )
            }
            selected = [match]
        } else {
            throw NSError(
                domain: "BsafeCapture", code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Invalid display filter: \(filter)"]
            )
        }
    }
    return (selected, content)
}

/// SCWindows for the given window-server ids. Fetched after the overlay windows exist, so a capturable
/// overlay can be excluded from our own capture (no feedback loop).
func shareableWindows(withIDs ids: Set<CGWindowID>) -> [SCWindow] {
    if ids.isEmpty { return [] }
    let semaphore = DispatchSemaphore(value: 0)
    var found: [SCWindow] = []
    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: false) { content, _ in
        found = content?.windows.filter { ids.contains($0.windowID) } ?? []
        semaphore.signal()
    }
    // Best effort on timeout (5 s): return what we have so the restart queue
    // never blocks forever. The fetchDisplay/startCapture timeouts below are
    // the ones counted as restart failures.
    if semaphore.wait(timeout: .now() + 5) == .timedOut {
        return found
    }
    return found
}

/// When set, overlay windows are capturable and our own app is excluded from capture instead.
func isOverlayCapturable() -> Bool {
    return ProcessInfo.processInfo.environment["BSAFE_OVERLAY_CAPTURABLE"] == "1"
}

/// Overlay present-lead in seconds (BSAFE_PRESENT_LEAD_MS env, default 16 ms).
func presentLeadSeconds() -> Double {
    if let s = ProcessInfo.processInfo.environment["BSAFE_PRESENT_LEAD_MS"], let ms = Double(s) {
        return ms / 1000.0
    }
    return 0.016
}

/// Recompute the excluded windows the same way startup does, for stream
/// restarts (displays/windows can be re-created after sleep/lock).
/// When the overlay is not capturable-excluded this returns [] without
/// touching AppKit at all (callers must also avoid snapshotting window ids
/// in that case — see snapshotOverlayWindowIDs).
func computeExcludedWindows(capturable: Bool, overlayWindowIDs: Set<CGWindowID>) -> [SCWindow] {
    guard capturable else { return [] }
    return shareableWindows(withIDs: overlayWindowIDs)
}

/// Synchronously fetch the SCDisplay with the given displayID, or nil when
/// gone. A 5 s timeout also returns nil so the restart queue never blocks
/// forever; the caller counts nil as a restart failure.
func fetchDisplay(displayID: UInt32) -> SCDisplay? {
    var fetched: SCShareableContent?
    let sem = DispatchSemaphore(value: 0)
    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, _ in
        fetched = content
        sem.signal()
    }
    if sem.wait(timeout: .now() + 5) == .timedOut {
        return nil
    }
    return fetched?.displays.first(where: { $0.displayID == displayID })
}

/// Backoff delay for restart attempt N (1-based): 0.25, 0.5, 1, 2, then 3 s.
func restartDelay(forAttempt attempt: Int) -> Double {
    switch attempt {
    case 1: return 0.25
    case 2: return 0.5
    case 3: return 1.0
    case 4: return 2.0
    default: return 3.0
    }
}
