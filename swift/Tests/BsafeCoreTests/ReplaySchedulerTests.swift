import XCTest
@testable import BsafeCore

final class ReplaySchedulerTests: XCTestCase {
    func testFastDetectCreditPipeline() {
        // detect 10ms, overhead 0, 60fps (pts = k/60).
        var s = ReplayScheduler(
            detectS: Array(repeating: 0.010, count: 10), overheadS: 0)
        XCTAssertEqual(s.beginFrame(k: 0, pts: 0), [])
        s.frameIngested(k: 0, pts: 0)
        XCTAssertEqual(s.requests, 1)

        let pts1 = 1.0 / 60.0
        XCTAssertEqual(s.beginFrame(k: 1, pts: pts1), [0])
        // Newest ingested (0) was already requested -> credit pending.
        XCTAssertTrue(s.creditPending)
        s.frameIngested(k: 1, pts: pts1)
        XCTAssertFalse(s.creditPending)
        XCTAssertEqual(s.requests, 2)
        XCTAssertEqual(s.completed, 1)
        XCTAssertEqual(s.meanLatency, 0.010, accuracy: 1e-9)

        // Request for frame 1 issued at pts1 completes at pts1+0.010; it must
        // be returned by beginFrame(2) at 2/60.
        let pts2 = 2.0 / 60.0
        XCTAssertEqual(s.beginFrame(k: 2, pts: pts2), [1])
        XCTAssertEqual(s.completed, 2)
    }

    func testSlowDetectSkipsFrames() {
        // detect 45ms, overhead 0, 50fps (pts = k*0.02).
        var s = ReplayScheduler(
            detectS: Array(repeating: 0.045, count: 10), overheadS: 0)
        // Frame 0 requested at 0, completes at 0.045.
        XCTAssertEqual(s.beginFrame(k: 0, pts: 0), [])
        s.frameIngested(k: 0, pts: 0)
        XCTAssertEqual(s.requests, 1)

        XCTAssertEqual(s.beginFrame(k: 1, pts: 0.02), [])
        s.frameIngested(k: 1, pts: 0.02)
        XCTAssertEqual(s.requests, 1)

        // 0.045 > 0.04 so nothing completes yet.
        XCTAssertEqual(s.beginFrame(k: 2, pts: 0.04), [])
        s.frameIngested(k: 2, pts: 0.04)
        XCTAssertEqual(s.requests, 1)

        // Completes 0.045 <= 0.06: returns [0]; newest ingested is 2, which was
        // never requested, so request 2 issued at 0.045 (completes 0.090).
        XCTAssertEqual(s.beginFrame(k: 3, pts: 0.06), [0])
        XCTAssertFalse(s.creditPending)
        XCTAssertEqual(s.requests, 2)
        s.frameIngested(k: 3, pts: 0.06)
        XCTAssertEqual(s.requests, 2)

        XCTAssertEqual(s.beginFrame(k: 4, pts: 0.08), [])
        s.frameIngested(k: 4, pts: 0.08)

        // 0.090 <= 0.10: returns [2]; newest ingested is 4 -> request 4 at 0.090.
        XCTAssertEqual(s.beginFrame(k: 5, pts: 0.10), [2])
        XCTAssertEqual(s.requests, 3)
        s.frameIngested(k: 5, pts: 0.10)
        XCTAssertEqual(s.requests, 3)

        XCTAssertEqual(s.requests, 3)
        XCTAssertEqual(s.completed, 2)
        // Latencies: (0.045 - 0.0) + (0.090 - 0.04) = 0.095; mean 0.0475.
        XCTAssertEqual(s.latencySum, 0.095, accuracy: 1e-9)
        XCTAssertEqual(s.meanLatency, 0.0475, accuracy: 1e-9)
    }

    func testOverheadAddsToLatency() {
        var s = ReplayScheduler(detectS: [0.010, 0.010, 0.010], overheadS: 0.006)
        XCTAssertEqual(s.beginFrame(k: 0, pts: 0), [])
        s.frameIngested(k: 0, pts: 0) // completes at 0.016
        // Frame spacing 1/60: 0.016 <= 0.0167 so [0] at k=1.
        XCTAssertEqual(s.beginFrame(k: 1, pts: 1.0 / 60.0), [0])
        XCTAssertEqual(s.latencySum, 0.016, accuracy: 1e-9)
        XCTAssertEqual(s.meanLatency, 0.016, accuracy: 1e-9)
        XCTAssertEqual(s.requests, 1)
        XCTAssertEqual(s.completed, 1)
    }

    func testBeginFrameTimedReturnsCompletionTimes() {
        var s = ReplayScheduler(
            detectS: Array(repeating: 0.010, count: 10), overheadS: 0.006)
        XCTAssertTrue(s.beginFrameTimed(k: 0, pts: 0).isEmpty)
        s.frameIngested(k: 0, pts: 0) // completes at 0.016
        let pts1 = 1.0 / 60.0
        let timed = s.beginFrameTimed(k: 1, pts: pts1)
        XCTAssertEqual(timed.count, 1)
        XCTAssertEqual(timed[0].frame, 0)
        XCTAssertEqual(timed[0].completion, 0.016, accuracy: 1e-9)
        // beginFrame delegates to beginFrameTimed: same completion reported once.
        var s2 = ReplayScheduler(
            detectS: Array(repeating: 0.010, count: 10), overheadS: 0.006)
        XCTAssertTrue(s2.beginFrame(k: 0, pts: 0).isEmpty)
        s2.frameIngested(k: 0, pts: 0)
        XCTAssertEqual(s2.beginFrame(k: 1, pts: pts1), [0])
    }
}
