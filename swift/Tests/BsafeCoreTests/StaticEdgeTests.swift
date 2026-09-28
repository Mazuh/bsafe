import Foundation
import XCTest
@testable import BsafeCore

// MARK: - Composites

/// Frame whose top band (or left band) is static texture `s` and whose remainder
/// is texture `m` scrolled up by `scroll` px: out[x, y] = m[x, y + scroll] there.
/// Moving values are contrast-reduced by `contrastDiv`. Output size == s.size.
private func composite(`static` s: LumaImage, moving m: LumaImage, scroll: Int, staticRowsBelow: Int?,
                       staticColsBelow: Int?, contrastDiv: Int = 1) -> LumaImage {
    let w = s.width
    let h = s.height
    var out = [UInt8](repeating: 128, count: w * h)
    for y in 0..<h {
        let my = y + scroll
        let rowOut = y * w
        for x in 0..<w {
            let isStatic = (staticRowsBelow != nil && y < staticRowsBelow!)
                || (staticColsBelow != nil && x < staticColsBelow!)
            if isStatic {
                out[rowOut + x] = s[x, y]
            } else if x < m.width && my >= 0 && my < m.height {
                let v = Int(m[x, my])
                out[rowOut + x] = UInt8(128 + (v - 128) / contrastDiv)
            } else {
                out[rowOut + x] = 128
            }
        }
    }
    return LumaImage(width: w, height: h, pixels: out)
}

/// Frame whose left band (x < staticColsBelow) is static texture `s` and whose
/// right remainder is texture `m` moved DOWN by `shiftDown` px: out[x, y] = m[x, y - shiftDown].
private func compositeMovingDown(`static` s: LumaImage, moving m: LumaImage, shiftDown: Int,
                                 staticColsBelow: Int) -> LumaImage {
    let w = s.width
    let h = s.height
    var out = [UInt8](repeating: 128, count: w * h)
    for y in 0..<h {
        let my = y - shiftDown
        let rowOut = y * w
        for x in 0..<w {
            if x < staticColsBelow {
                out[rowOut + x] = s[x, y]
            } else if x < m.width && my >= 0 && my < m.height {
                out[rowOut + x] = m[x, my]
            } else {
                out[rowOut + x] = 128
            }
        }
    }
    return LumaImage(width: w, height: h, pixels: out)
}

// MARK: - Static-edge regression tests

/// A box whose sample region is partly a STATIC textured band must still follow the
/// content scrolling inside it. The current estimator pins such boxes at shift 0.
final class StaticEdgeTests: XCTestCase {

    // A
    func testStaticBandDoesNotPinMovingContent() {
        let staticImg = makeTexture(width: 640, height: 400, seed: 0x51)
        let movingImg = makeTexture(width: 640, height: 600, seed: 0x52)
        let prev = pyr(
            composite(`static`: staticImg, moving: movingImg, scroll: 0,
                      staticRowsBelow: 150, staticColsBelow: nil),
            downscale: 1)
        let cur = pyr(
            composite(`static`: staticImg, moving: movingImg, scroll: 10,
                      staticRowsBelow: 150, staticColsBelow: nil),
            downscale: 1)
        let box = TrackBox(x: 200, y: 130, w: 120, h: 100)
        let s = MotionEstimator.estimate(
            from: prev, to: cur, box: box, predicted: Shift.zero, searchRadius: 12)
        XCTAssertNotNil(s, "expected shift dx~0 dy~-10 past the static band y<150, got nil")
        if let s = s {
            XCTAssertLessThanOrEqual(abs(s.dx - 0), 1.0, "dx=\(s.dx) dy=\(s.dy) expected dx=0")
            XCTAssertLessThanOrEqual(abs(s.dy - (-10)), 1.0, "dy=\(s.dy) dx=\(s.dx) expected dy=-10")
        }
    }

    // A2: same, but the moving content is low contrast while the static band is full contrast.
    func testStaticBandDoesNotPinLowContrastMovingContent() {
        let staticImg = makeTexture(width: 640, height: 400, seed: 0x51)
        let movingImg = makeTexture(width: 640, height: 600, seed: 0x52)
        let prev = pyr(
            composite(`static`: staticImg, moving: movingImg, scroll: 0,
                      staticRowsBelow: 150, staticColsBelow: nil, contrastDiv: 4),
            downscale: 1)
        let cur = pyr(
            composite(`static`: staticImg, moving: movingImg, scroll: 10,
                      staticRowsBelow: 150, staticColsBelow: nil, contrastDiv: 4),
            downscale: 1)
        let box = TrackBox(x: 200, y: 130, w: 120, h: 100)
        let s = MotionEstimator.estimate(
            from: prev, to: cur, box: box, predicted: Shift.zero, searchRadius: 12)
        XCTAssertNotNil(s, "expected shift dx~0 dy~-10 with contrastDiv=4, got nil")
        if let s = s {
            XCTAssertLessThanOrEqual(abs(s.dx - 0), 1.0, "dx=\(s.dx) dy=\(s.dy) expected dx=0")
            XCTAssertLessThanOrEqual(abs(s.dy - (-10)), 1.0, "dy=\(s.dy) dx=\(s.dx) expected dy=-10")
        }
    }

    // A3: same behaviour on a downscaled pyramid. The image is pyramid level 0 (capture / 3): static rows
    // y < 75 are capture y < 225, and a 4 px scroll at level 0 is -12 capture px.
    func testStaticBandDoesNotPinMovingContentDownscale3() {
        let staticImg = makeTexture(width: 960, height: 600, seed: 0x51)
        let movingImg = makeTexture(width: 960, height: 800, seed: 0x52)
        let prev = pyr(
            composite(`static`: staticImg, moving: movingImg, scroll: 0,
                      staticRowsBelow: 75, staticColsBelow: nil),
            downscale: 3)
        let cur = pyr(
            composite(`static`: staticImg, moving: movingImg, scroll: 4,
                      staticRowsBelow: 75, staticColsBelow: nil),
            downscale: 3)
        let box = TrackBox(x: 300, y: 195, w: 180, h: 150)
        let s = MotionEstimator.estimate(
            from: prev, to: cur, box: box, predicted: Shift.zero, searchRadius: 12)
        XCTAssertNotNil(s, "expected shift dx~0 dy~-12 at downscale 3, got nil")
        if let s = s {
            XCTAssertLessThanOrEqual(abs(s.dx - 0), 3.0, "dx=\(s.dx) dy=\(s.dy) expected dx=0")
            XCTAssertLessThanOrEqual(abs(s.dy - (-12)), 3.0, "dy=\(s.dy) dx=\(s.dx) expected dy=-12")
        }
    }

    // B: a box over static content must not follow the moving content beside it.
    func testStaticTargetIgnoresMovingSurroundings() {
        let staticImg = makeTexture(width: 640, height: 400, seed: 0x51)
        let movingImg = makeTexture(width: 640, height: 400, seed: 0x52)
        let prev = pyr(staticImg, downscale: 1)
        let cur = pyr(
            compositeMovingDown(`static`: staticImg, moving: movingImg,
                                shiftDown: 12, staticColsBelow: 340),
            downscale: 1)
        let boxes: [(String, TrackBox)] = [
            ("boxA", TrackBox(x: 200, y: 150, w: 130, h: 100)),
            ("boxB", TrackBox(x: 220, y: 150, w: 140, h: 100)),
        ]
        for (name, box) in boxes {
            let s = MotionEstimator.estimate(
                from: prev, to: cur, box: box, predicted: Shift.zero, searchRadius: 12)
            guard let s = s else {
                XCTFail("\(name) must return Shift.zero, got nil (moving columns x>=340 drift down 12)")
                continue
            }
            XCTAssertTrue(s.dx == 0 && s.dy == 0, "\(name) returned dx=\(s.dx) dy=\(s.dy), expected 0/0")
        }
    }

    // C: a tracked box must keep following a scroll as it enters the static band.
    func testTrackerFollowsScrollNearStaticBand() {
        let staticImg = makeTexture(width: 640, height: 400, seed: 0x51)
        let movingImg = makeTexture(width: 640, height: 600, seed: 0x52)
        func frame(_ i: Int) -> LumaImage {
            composite(`static`: staticImg, moving: movingImg, scroll: 10 * i,
                      staticRowsBelow: 150, staticColsBelow: nil)
        }
        let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400)
        tracker.ingestFrame(seq: 0, pts: 0, pyramid: pyr(frame(0), downscale: 1))
        let adopted = tracker.applyDetections(
            seq: 0, boxes: [TrackBox(x: 200, y: 250, w: 120, h: 100)])
        XCTAssertTrue(adopted, "applyDetections(seq: 0) returned false, track count=\(tracker.tracks.count)")
        for i in 1...14 {
            tracker.ingestFrame(seq: UInt32(i), pts: Double(i) / 60.0,
                                pyramid: pyr(frame(i), downscale: 1))
            XCTAssertEqual(tracker.tracks.count, 1, "frame \(i): track count=\(tracker.tracks.count)")
            guard let t = tracker.tracks.first else { continue }
            let expected = 250.0 - 10.0 * Double(i)
            XCTAssertLessThanOrEqual(
                abs(t.box.y - expected), 3.0,
                "frame \(i): box.y=\(t.box.y) expected \(expected)")
        }
    }
}
