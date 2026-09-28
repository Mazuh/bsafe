import XCTest
@testable import BsafeCore

final class BlendTests: XCTestCase {
    let b = TrackBox(x: 50, y: 50, w: 60, h: 50)

    func testGrowingEdgesSnap() {
        let base = makeTexture(width: 320, height: 200, seed: 0xB1E4D1)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200, config: TrackerConfig(shrinkAlpha: 0.25))
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1, "after first apply count=\(tr.tracks.count)")
        let grown = TrackBox(x: 40, y: 45, w: 80, h: 60)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [grown]))
        XCTAssertEqual(tr.tracks.count, 1, "after grow apply count=\(tr.tracks.count)")
        let box = tr.tracks[0].box
        XCTAssertEqual(box.x, 40, accuracy: 1e-9, "x=\(box.x)")
        XCTAssertEqual(box.y, 45, accuracy: 1e-9, "y=\(box.y)")
        XCTAssertEqual(box.w, 80, accuracy: 1e-9, "w=\(box.w)")
        XCTAssertEqual(box.h, 60, accuracy: 1e-9, "h=\(box.h)")
    }

    func testShrinkingEdgeMovesByShrinkAlpha() {
        let base = makeTexture(width: 320, height: 200, seed: 0xB1E4D2)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200, config: TrackerConfig(shrinkAlpha: 0.25))
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1, "after first apply count=\(tr.tracks.count)")
        let narrow = TrackBox(x: 50, y: 50, w: 40, h: 50)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [narrow]))
        XCTAssertEqual(tr.tracks.count, 1, "after narrow apply count=\(tr.tracks.count)")
        var box = tr.tracks[0].box
        XCTAssertEqual(box.x, 50, accuracy: 1e-9, "x=\(box.x)")
        XCTAssertEqual(box.w, 55, accuracy: 1e-9, "w=\(box.w)")
        XCTAssertEqual(box.y, 50, accuracy: 1e-9, "y=\(box.y)")
        XCTAssertEqual(box.h, 50, accuracy: 1e-9, "h=\(box.h)")
        for _ in 0..<3 {
            XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [narrow]))
            XCTAssertEqual(tr.tracks.count, 1, "repeated narrow count=\(tr.tracks.count)")
        }
        box = tr.tracks[0].box
        let expected = 40 + 20 * pow(0.75, 4.0)
        XCTAssertEqual(box.w, expected, accuracy: 1e-9, "w=\(box.w) expected=\(expected)")
    }

    func testNarrowFlickerKeepsCoverageThenRecovers() {
        let base = makeTexture(width: 320, height: 200, seed: 0xB1E4D3)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200, config: TrackerConfig(shrinkAlpha: 0.25))
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1, "after first apply count=\(tr.tracks.count)")
        let flicker = TrackBox(x: 50, y: 50, w: 30, h: 50)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [flicker]))
        XCTAssertEqual(tr.tracks.count, 1, "after flicker count=\(tr.tracks.count)")
        var box = tr.tracks[0].box
        let right = box.x + box.w
        XCTAssertTrue(right >= 102.5, "right=\(right) must be >= 102.5")
        XCTAssertEqual(right, 102.5, accuracy: 1e-9, "right=\(right)")
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1, "after recover count=\(tr.tracks.count)")
        box = tr.tracks[0].box
        XCTAssertEqual(box.x, b.x, accuracy: 1e-9, "x=\(box.x)")
        XCTAssertEqual(box.y, b.y, accuracy: 1e-9, "y=\(box.y)")
        XCTAssertEqual(box.w, b.w, accuracy: 1e-9, "w=\(box.w)")
        XCTAssertEqual(box.h, b.h, accuracy: 1e-9, "h=\(box.h)")
    }

    func testVerticalEdges() {
        let base = makeTexture(width: 320, height: 200, seed: 0xB1E4D4)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200, config: TrackerConfig(shrinkAlpha: 0.25))
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1, "after first apply count=\(tr.tracks.count)")
        let shifted = TrackBox(x: 50, y: 40, w: 60, h: 40)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [shifted]))
        XCTAssertEqual(tr.tracks.count, 1, "after vertical apply count=\(tr.tracks.count)")
        let box = tr.tracks[0].box
        XCTAssertEqual(box.y, 40, accuracy: 1e-9, "y=\(box.y)")
        XCTAssertEqual(box.y + box.h, 95, accuracy: 1e-9, "bottom=\(box.y + box.h)")
        XCTAssertEqual(box.h, 55, accuracy: 1e-9, "h=\(box.h)")
    }

    func testShrinkAlphaConfig() {
        let base = makeTexture(width: 320, height: 200, seed: 0xB1E4D5)
        let tr = DisplayTracker(
            frameWidth: 320, frameHeight: 200,
            config: TrackerConfig(persistPasses: 8, shrinkAlpha: 1.0))
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1, "after first apply count=\(tr.tracks.count)")
        let narrow = TrackBox(x: 50, y: 50, w: 40, h: 50)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [narrow]))
        XCTAssertEqual(tr.tracks.count, 1, "after narrow count=\(tr.tracks.count)")
        let box = tr.tracks[0].box
        XCTAssertEqual(box.w, 40, accuracy: 1e-9, "w=\(box.w)")
        XCTAssertEqual(TrackerConfig(shrinkAlpha: -1).shrinkAlpha, 0, accuracy: 1e-9, "clamped=\(TrackerConfig(shrinkAlpha: -1).shrinkAlpha)")
        XCTAssertEqual(TrackerConfig(shrinkAlpha: 2).shrinkAlpha, 1, accuracy: 1e-9, "clamped=\(TrackerConfig(shrinkAlpha: 2).shrinkAlpha)")
    }
}
