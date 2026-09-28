import XCTest
@testable import BsafeCore

final class CoreBoxTests: XCTestCase {
    /// Copy of the static background with the moving patch pasted at (260, 280 - 10*i).
    /// Pixels falling outside the frame are skipped.
    private func movingPatchFrame(bg: LumaImage, patch: LumaImage, index i: Int) -> LumaImage {
        var pixels = bg.pixels
        let y0 = 280 - 10 * i
        for py in 0..<patch.height {
            let y = y0 + py
            if y < 0 || y >= bg.height { continue }
            for px in 0..<patch.width {
                let x = 260 + px
                if x < 0 || x >= bg.width { continue }
                pixels[y * bg.width + x] = patch.pixels[py * patch.width + px]
            }
        }
        return LumaImage(width: bg.width, height: bg.height, pixels: pixels)
    }

    func testOversizedDetectionDoesNotPinMotion() {
        let bg = makeTexture(width: 640, height: 400, seed: 0x77)
        let patch = makeTexture(width: 120, height: 100, seed: 0x78)
        for scale in [1.5, 2.0] {
            let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400)
            tracker.ingestFrame(seq: 0, pts: 0, pyramid: pyr(movingPatchFrame(bg: bg, patch: patch, index: 0)))
            XCTAssertTrue(
                tracker.applyDetections(seq: 0, boxes: [TrackBox(x: 260, y: 280, w: 120, h: 100)]),
                "scale \(scale): seq-0 tight detection must apply")
            for i in 1...3 {
                tracker.ingestFrame(
                    seq: UInt32(i), pts: Double(i) / 60.0,
                    pyramid: pyr(movingPatchFrame(bg: bg, patch: patch, index: i)))
            }
            let w = 120 * scale
            let h = 100 * scale
            let det = TrackBox(x: 320 - w / 2.0, y: 300 - h / 2.0, w: w, h: h)
            XCTAssertTrue(
                tracker.applyDetections(seq: 3, boxes: [det]),
                "scale \(scale): oversized detection det=\(det) must apply")
            for i in 4...12 {
                tracker.ingestFrame(
                    seq: UInt32(i), pts: Double(i) / 60.0,
                    pyramid: pyr(movingPatchFrame(bg: bg, patch: patch, index: i)))
                XCTAssertEqual(
                    tracker.tracks.count, 1,
                    "scale \(scale) frame \(i): tracks.count=\(tracker.tracks.count)")
                guard tracker.tracks.count == 1 else { continue }
                let tr = tracker.tracks[0]
                XCTAssertLessThanOrEqual(
                    abs(tr.velocity.dy - (-600)), 60.0,
                    "scale \(scale) frame \(i): velocity.dy=\(tr.velocity.dy)")
                let patchY = Double(280 - 10 * i)
                XCTAssertLessThanOrEqual(
                    tr.box.y, patchY + 1,
                    "scale \(scale) frame \(i): box.y=\(tr.box.y) patchY=\(patchY)")
                XCTAssertGreaterThanOrEqual(
                    tr.box.y + tr.box.h, patchY + 99,
                    "scale \(scale) frame \(i): box.bottom=\(tr.box.y + tr.box.h) patchY=\(patchY)")
                XCTAssertLessThanOrEqual(
                    abs(tr.core.h - 100 * scale), 3.0,
                    "scale \(scale) frame \(i): core.h=\(tr.core.h) expected=\(100 * scale)")
            }
        }
    }

    func testCoreFollowsTightDetectionBoxGrows() {
        let bg = makeTexture(width: 640, height: 400, seed: 0x77)
        let patch = makeTexture(width: 120, height: 100, seed: 0x78)
        let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400)
        tracker.ingestFrame(seq: 0, pts: 0, pyramid: pyr(movingPatchFrame(bg: bg, patch: patch, index: 0)))
        XCTAssertTrue(
            tracker.applyDetections(seq: 0, boxes: [TrackBox(x: 260, y: 280, w: 120, h: 100)]),
            "seq-0 tight detection must apply")
        for i in 1...3 {
            tracker.ingestFrame(
                seq: UInt32(i), pts: Double(i) / 60.0,
                pyramid: pyr(movingPatchFrame(bg: bg, patch: patch, index: i)))
        }
        let det = TrackBox(x: 260, y: 250, w: 160, h: 100)
        XCTAssertTrue(tracker.applyDetections(seq: 3, boxes: [det]), "right-wider detection must apply")
        XCTAssertEqual(tracker.tracks.count, 1, "tracks.count=\(tracker.tracks.count)")
        guard tracker.tracks.count == 1 else { return }
        let tr = tracker.tracks[0]
        XCTAssertEqual(tr.core.x, det.x, accuracy: 1e-9, "core.x=\(tr.core.x)")
        XCTAssertEqual(tr.core.y, det.y, accuracy: 1e-9, "core.y=\(tr.core.y)")
        XCTAssertEqual(tr.core.w, det.w, accuracy: 1e-9, "core.w=\(tr.core.w)")
        XCTAssertEqual(tr.core.h, det.h, accuracy: 1e-9, "core.h=\(tr.core.h)")
        XCTAssertEqual(tr.box.x + tr.box.w, 420, accuracy: 1e-9, "box.right=\(tr.box.x + tr.box.w)")
        XCTAssertLessThanOrEqual(abs(tr.box.y - 250), 1.0, "box.y=\(tr.box.y)")
        XCTAssertLessThanOrEqual(abs(tr.box.x - 260), 1.0, "box.x=\(tr.box.x)")
        XCTAssertLessThanOrEqual(tr.box.x, tr.core.x + 1, "box.x=\(tr.box.x) core.x=\(tr.core.x)")
        XCTAssertLessThanOrEqual(tr.box.y, tr.core.y + 1, "box.y=\(tr.box.y) core.y=\(tr.core.y)")
        XCTAssertGreaterThanOrEqual(
            tr.box.x + tr.box.w, tr.core.x + tr.core.w - 1,
            "box.right=\(tr.box.x + tr.box.w) core.right=\(tr.core.x + tr.core.w)")
        XCTAssertGreaterThanOrEqual(
            tr.box.y + tr.box.h, tr.core.y + tr.core.h - 1,
            "box.bottom=\(tr.box.y + tr.box.h) core.bottom=\(tr.core.y + tr.core.h)")
    }

    func testStaleOversizedDetectionCatchesUp() {
        let bg = makeTexture(width: 640, height: 400, seed: 0x77)
        let patch = makeTexture(width: 120, height: 100, seed: 0x78)
        let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400)
        tracker.ingestFrame(seq: 0, pts: 0, pyramid: pyr(movingPatchFrame(bg: bg, patch: patch, index: 0)))
        XCTAssertTrue(
            tracker.applyDetections(seq: 0, boxes: [TrackBox(x: 260, y: 280, w: 120, h: 100)]),
            "seq-0 tight detection must apply")
        for i in 1...5 {
            tracker.ingestFrame(
                seq: UInt32(i), pts: Double(i) / 60.0,
                pyramid: pyr(movingPatchFrame(bg: bg, patch: patch, index: i)))
        }
        let stale = TrackBox(x: 260 - 48, y: 250 - 40, w: 120 * 1.8, h: 100 * 1.8)
        XCTAssertTrue(
            tracker.applyDetections(seq: 3, boxes: [stale]),
            "stale padded detection stale=\(stale) must apply")
        XCTAssertEqual(tracker.tracks.count, 1, "tracks.count=\(tracker.tracks.count)")
        guard tracker.tracks.count == 1 else { return }
        let tr = tracker.tracks[0]
        XCTAssertLessThanOrEqual(
            abs(tr.core.y - 190), 3.0,
            "core.y=\(tr.core.y) expected~190")
        XCTAssertLessThanOrEqual(
            tr.box.y, 230 + 1,
            "box.y=\(tr.box.y) patchY=230")
        XCTAssertGreaterThanOrEqual(
            tr.box.y + tr.box.h, 230 + 99,
            "box.bottom=\(tr.box.y + tr.box.h) patchY=230")
        for i in 6...12 {
            tracker.ingestFrame(
                seq: UInt32(i), pts: Double(i) / 60.0,
                pyramid: pyr(movingPatchFrame(bg: bg, patch: patch, index: i)))
            XCTAssertEqual(
                tracker.tracks.count, 1,
                "frame \(i): tracks.count=\(tracker.tracks.count)")
            guard tracker.tracks.count == 1 else { continue }
            let t = tracker.tracks[0]
            XCTAssertLessThanOrEqual(
                abs(t.velocity.dy - (-600)), 60.0,
                "frame \(i): velocity.dy=\(t.velocity.dy)")
            let patchY = Double(280 - 10 * i)
            XCTAssertLessThanOrEqual(
                t.box.y, patchY + 1,
                "frame \(i): box.y=\(t.box.y) patchY=\(patchY)")
            XCTAssertGreaterThanOrEqual(
                t.box.y + t.box.h, patchY + 99,
                "frame \(i): box.bottom=\(t.box.y + t.box.h) patchY=\(patchY)")
        }
    }

    func testShrinkAlphaEdgeCases() {
        XCTAssertEqual(
            TrackerConfig(shrinkAlpha: .nan).shrinkAlpha, 0.1, accuracy: 1e-9,
            "nan shrinkAlpha clamped to \(TrackerConfig(shrinkAlpha: .nan).shrinkAlpha)")
        let base = makeTexture(width: 320, height: 200, seed: 0xC0E)
        let tr = DisplayTracker(
            frameWidth: 320, frameHeight: 200,
            config: TrackerConfig(persistPasses: 8, shrinkAlpha: 0))
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        let b = TrackBox(x: 50, y: 50, w: 60, h: 50)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]), "first detection must apply")
        let narrow = TrackBox(x: 50, y: 50, w: 40, h: 50)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [narrow]), "narrow detection must apply")
        XCTAssertEqual(tr.tracks.count, 1, "tracks.count=\(tr.tracks.count)")
        guard tr.tracks.count == 1 else { return }
        let box = tr.tracks[0].box
        XCTAssertEqual(box.x, 50, accuracy: 1e-9, "box.x=\(box.x)")
        XCTAssertEqual(box.y, 50, accuracy: 1e-9, "box.y=\(box.y)")
        XCTAssertEqual(box.w, 60, accuracy: 1e-9, "box.w=\(box.w)")
        XCTAssertEqual(box.h, 50, accuracy: 1e-9, "box.h=\(box.h)")
        let core = tr.tracks[0].core
        XCTAssertEqual(core.x, 50, accuracy: 1e-9, "core.x=\(core.x)")
        XCTAssertEqual(core.y, 50, accuracy: 1e-9, "core.y=\(core.y)")
        XCTAssertEqual(core.w, 40, accuracy: 1e-9, "core.w=\(core.w)")
        XCTAssertEqual(core.h, 50, accuracy: 1e-9, "core.h=\(core.h)")
    }
}
