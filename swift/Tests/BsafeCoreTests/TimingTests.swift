import XCTest
@testable import BsafeCore

final class TimingTests: XCTestCase {
    /// A long inter-frame gap must clamp dt to 0.1 s so velocity is not
    /// overestimated. New track starts with velocity 0, so the blend gives
    /// 0.7 * (-20 / 0.1) + 0.3 * 0 = -140. Key property: |velocity.dy| must
    /// be far below the unclamped value 0.7 * (20 / (1/60)) = 840.
    func testLongFrameGapClampsVelocity() {
        let base = makeTexture(width: 640, height: 400, seed: 0x44444)
        let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400)
        tracker.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(
            tracker.applyDetections(seq: 0, boxes: [TrackBox(x: 200, y: 150, w: 160, h: 100)]),
            "applyDetections(seq: 0) must return true"
        )
        tracker.ingestFrame(seq: 1, pts: 0.15, pyramid: pyr(translated(base, dx: 0, dy: -20)))
        XCTAssertEqual(tracker.tracks.count, 1, "expected 1 track, got \(tracker.tracks.count)")
        let track = tracker.tracks[0]
        XCTAssertLessThanOrEqual(
            abs(track.box.y - 130), 2.0,
            "box.y=\(track.box.y), expected within 2 of 130"
        )
        XCTAssertGreaterThanOrEqual(
            track.velocity.dy, -190.0,
            "velocity.dy=\(track.velocity.dy), expected >= -190"
        )
        XCTAssertLessThanOrEqual(
            track.velocity.dy, -90.0,
            "velocity.dy=\(track.velocity.dy), expected <= -90"
        )
    }
}
