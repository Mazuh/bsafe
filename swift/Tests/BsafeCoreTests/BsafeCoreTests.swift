import Foundation
import XCTest
@testable import BsafeCore

// MARK: - Helpers

func makeTexture(width: Int, height: Int, seed: UInt64) -> LumaImage {
    var state: UInt64 = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    func next() -> UInt64 {
        state ^= state &<< 13
        state ^= state &>> 7
        state ^= state &<< 17
        return state
    }
    var noise = [UInt8](repeating: 0, count: width * height)
    for i in 0..<noise.count {
        noise[i] = UInt8(truncatingIfNeeded: next() & 0xFF)
    }
    var out = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
        for x in 0..<width {
            var sum = 0
            var cnt = 0
            for dy in -1...1 {
                for dx in -1...1 {
                    let nx = x + dx
                    let ny = y + dy
                    if nx >= 0 && nx < width && ny >= 0 && ny < height {
                        sum += Int(noise[ny * width + nx])
                        cnt += 1
                    }
                }
            }
            out[y * width + x] = UInt8((sum + cnt / 2) / cnt)
        }
    }
    return LumaImage(width: width, height: height, pixels: out)
}

func translated(_ img: LumaImage, dx: Int, dy: Int, seed: UInt64 = 0xDEADBEEF) -> LumaImage {
    var state: UInt64 = seed
    func nextFill() -> UInt64 {
        state ^= state &<< 13
        state ^= state &>> 7
        state ^= state &<< 17
        return state
    }
    var out = [UInt8](repeating: 0, count: img.width * img.height)
    // Pre-generate fill stream deterministically in row-major order of uncovered pixels
    // by simply consuming RNG as we go (deterministic given dx,dy and dims).
    for y in 0..<img.height {
        for x in 0..<img.width {
            let sx = x - dx
            let sy = y - dy
            if sx >= 0 && sx < img.width && sy >= 0 && sy < img.height {
                out[y * img.width + x] = img.pixels[sy * img.width + sx]
            } else {
                out[y * img.width + x] = UInt8(truncatingIfNeeded: nextFill() & 0xFF)
            }
        }
    }
    return LumaImage(width: img.width, height: img.height, pixels: out)
}

func pyr(_ img: LumaImage, downscale: Int = 1) -> LumaPyramid {
    LumaPyramid(level0: img, downscale: downscale)
}

final class BsafeCoreTests: XCTestCase {

    // 1. fromBGRA
    func testFromBGRA() {
        let w = 12
        let h = 6
        let bpr = 64 // padded (12*4=48)
        var buf = [UInt8](repeating: 0, count: h * bpr)
        // Top 3 rows red (B0 G0 R255), bottom 3 rows green (B0 G255 R0)
        for y in 0..<h {
            for x in 0..<w {
                let off = y * bpr + x * 4
                buf[off + 3] = 255 // A
                if y < 3 {
                    buf[off] = 0; buf[off + 1] = 0; buf[off + 2] = 255
                } else {
                    buf[off] = 0; buf[off + 1] = 255; buf[off + 2] = 0
                }
            }
        }
        let p1: LumaPyramid = buf.withUnsafeBytes { raw in
            LumaPyramid.fromBGRA(raw.baseAddress!, width: w, height: h, bytesPerRow: bpr)
        }
        XCTAssertEqual(p1.downscale, 1)
        XCTAssertEqual(p1.levels.count, 3)
        XCTAssertEqual(p1.levels[0].width, 12)
        XCTAssertEqual(p1.levels[0].height, 6)
        XCTAssertEqual(p1.levels[1].width, 6)
        XCTAssertEqual(p1.levels[1].height, 3)
        XCTAssertEqual(p1.levels[2].width, 3)
        XCTAssertEqual(p1.levels[2].height, 1)
        let red = Int(p1.levels[0][0, 0])
        let red2 = Int(p1.levels[0][11, 0])
        let grn = Int(p1.levels[0][0, 5])
        XCTAssertLessThanOrEqual(abs(red - 77), 2)
        XCTAssertLessThanOrEqual(abs(red2 - 77), 2)
        XCTAssertLessThanOrEqual(abs(grn - 150), 2)

        // d=2 via maxDimension
        let p2: LumaPyramid = buf.withUnsafeBytes { raw in
            LumaPyramid.fromBGRA(raw.baseAddress!, width: w, height: h, bytesPerRow: bpr, maxDimension: 6)
        }
        XCTAssertEqual(p2.downscale, 2)
        XCTAssertEqual(p2.levels[0].width, 6)
        XCTAssertEqual(p2.levels[0].height, 3)
        // row0 covers capture rows 0-1 (red) -> ~77; row1 covers 2-3 (mix) -> ~113; row2 covers 4-5 (green)
        let r0 = Int(p2.levels[0][0, 0])
        let r1 = Int(p2.levels[0][0, 1])
        let r2 = Int(p2.levels[0][0, 2])
        XCTAssertLessThanOrEqual(abs(r0 - 77), 3)
        XCTAssertLessThanOrEqual(abs(r2 - 150), 3)
        XCTAssertLessThanOrEqual(abs(r1 - 113), 5)

        // Pure colors
        let w2 = 4
        let h2 = 1
        let bpr2 = 16
        var buf2 = [UInt8](repeating: 0, count: h2 * bpr2)
        // px0 red, px1 green, px2 blue, px3 white
        let cols: [(UInt8, UInt8, UInt8)] = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
        for x in 0..<4 {
            let off = x * 4
            buf2[off] = cols[x].0
            buf2[off + 1] = cols[x].1
            buf2[off + 2] = cols[x].2
            buf2[off + 3] = 255
        }
        let p3: LumaPyramid = buf2.withUnsafeBytes { raw in
            LumaPyramid.fromBGRA(raw.baseAddress!, width: w2, height: h2, bytesPerRow: bpr2)
        }
        XCTAssertLessThanOrEqual(abs(Int(p3.levels[0][0, 0]) - 77), 2)
        XCTAssertLessThanOrEqual(abs(Int(p3.levels[0][1, 0]) - 150), 2)
        XCTAssertLessThanOrEqual(abs(Int(p3.levels[0][2, 0]) - 29), 2)
        XCTAssertLessThanOrEqual(abs(Int(p3.levels[0][3, 0]) - 255), 2)
    }

    // 2. estimate basic shifts
    func testEstimateShifts() {
        let base = makeTexture(width: 640, height: 400, seed: 0x12345)
        let prev = pyr(base, downscale: 1)
        let box = TrackBox(x: 200, y: 150, w: 120, h: 100)
        let cases: [(Int, Int)] = [(0, 0), (0, 17), (0, -40), (25, -9)]
        for (dx, dy) in cases {
            let curImg = translated(base, dx: dx, dy: dy)
            let cur = pyr(curImg, downscale: 1)
            let s = MotionEstimator.estimate(from: prev, to: cur, box: box, predicted: Shift.zero, searchRadius: 12)
            XCTAssertNotNil(s, "shift (\(dx),\(dy)) should be found")
            if let s = s {
                XCTAssertLessThanOrEqual(abs(s.dx - Double(dx)), 0.75, "dx for (\(dx),\(dy))")
                XCTAssertLessThanOrEqual(abs(s.dy - Double(dy)), 0.75, "dy for (\(dx),\(dy))")
            }
        }
        // (0,60) with predicted (0,55)
        do {
            let curImg = translated(base, dx: 0, dy: 60)
            let cur = pyr(curImg, downscale: 1)
            let s = MotionEstimator.estimate(
                from: prev, to: cur, box: box,
                predicted: Shift(dx: 0, dy: 55), searchRadius: 12)
            XCTAssertNotNil(s)
            if let s = s {
                XCTAssertLessThanOrEqual(abs(s.dx - 0), 0.75)
                XCTAssertLessThanOrEqual(abs(s.dy - 60), 0.75)
            }
        }
        // constant -> nil
        do {
            let c0 = LumaImage(width: 640, height: 400, pixels: [UInt8](repeating: 128, count: 640 * 400))
            let c1 = LumaImage(width: 640, height: 400, pixels: [UInt8](repeating: 128, count: 640 * 400))
            let s = MotionEstimator.estimate(
                from: pyr(c0), to: pyr(c1), box: box, predicted: Shift.zero, searchRadius: 12)
            XCTAssertNil(s, "constant image must be nil")
        }
        // independent noise -> nil
        do {
            let other = makeTexture(width: 640, height: 400, seed: 0x99999)
            let s = MotionEstimator.estimate(
                from: prev, to: pyr(other), box: box, predicted: Shift.zero, searchRadius: 12)
            XCTAssertNil(s, "independent noise must be nil")
        }
    }

    // 3. downscale 3
    func testEstimateDownscale3() {
        let base = makeTexture(width: 640, height: 400, seed: 0xABCDE)
        let prev = LumaPyramid(level0: base, downscale: 3)
        let curImg = translated(base, dx: 0, dy: 12)
        let cur = LumaPyramid(level0: curImg, downscale: 3)
        let box = TrackBox(x: 200, y: 150, w: 120, h: 100)
        let s = MotionEstimator.estimate(from: prev, to: cur, box: box, predicted: Shift.zero, searchRadius: 12)
        XCTAssertNotNil(s)
        if let s = s {
            XCTAssertLessThanOrEqual(abs(s.dx - 0), 2.25)
            XCTAssertLessThanOrEqual(abs(s.dy - 36), 2.25)
        }
    }

    // 4. tracker scroll
    func testTrackerScroll() {
        let base = makeTexture(width: 640, height: 400, seed: 0x11111)
        let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400)
        func frame(_ i: Int) -> LumaImage { translated(base, dx: 0, dy: i * 10) }
        tracker.ingestFrame(seq: 0, pts: 0, pyramid: pyr(frame(0)))
        let detBox = TrackBox(x: 200, y: 150, w: 120, h: 100)
        XCTAssertTrue(tracker.applyDetections(seq: 0, boxes: [detBox]))
        for i in 1...6 {
            tracker.ingestFrame(seq: UInt32(i), pts: Double(i) / 60.0, pyramid: pyr(frame(i)))
        }
        XCTAssertEqual(tracker.tracks.count, 1)
        let tb = tracker.tracks[0].box
        XCTAssertLessThanOrEqual(abs(tb.y - 210), 2.0, "box.y moved 60")
        XCTAssertLessThanOrEqual(abs(tracker.tracks[0].velocity.dy - 600), 60.0, "velocity")
        let rendered = tracker.renderBoxes(now: Double(6) / 60.0, presentLead: 0.05)
        XCTAssertEqual(rendered.count, 1)
        if let r = rendered.first {
            XCTAssertLessThanOrEqual(abs(r.y - tb.y), 1.0, "top edge")
            XCTAssertLessThanOrEqual(abs((r.y + r.h) - (tb.y + tb.h + 30)), 6.0, "bottom extended ~30")
        }
    }

    // 5. catch-up
    func testCatchUp() {
        let base = makeTexture(width: 640, height: 400, seed: 0x22222)
        let tracker = DisplayTracker(frameWidth: 640, frameHeight: 400)
        func frame(_ i: Int) -> LumaImage { translated(base, dx: 0, dy: i * 10) }
        for i in 0...4 {
            tracker.ingestFrame(seq: UInt32(i), pts: Double(i) / 60.0, pyramid: pyr(frame(i)))
        }
        let detBox = TrackBox(x: 200, y: 150, w: 120, h: 100)
        XCTAssertTrue(tracker.applyDetections(seq: 0, boxes: [detBox]))
        XCTAssertEqual(tracker.tracks.count, 1)
        let tb = tracker.tracks[0].box
        // frame-4 content is at y=150+40=190
        XCTAssertLessThanOrEqual(abs(tb.y - 190), 2.0)
        let v = tracker.tracks[0].velocity
        XCTAssertTrue(abs(v.dx) + abs(v.dy) > 1.0, "nonzero velocity")
    }

    // 6. persistence / eviction / alpha blend
    func testPersistenceAndEvictionAndBlend() {
        // persist 2 keeps after one miss, removes after two
        var cfg = TrackerConfig(persistPasses: 2, minPersistSeconds: 0)
        cfg.historyCapacity = 16
        let base = makeTexture(width: 320, height: 200, seed: 0x33333)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200, config: cfg)
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        let b = TrackBox(x: 50, y: 50, w: 60, h: 50)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: []))
        XCTAssertEqual(tr.tracks.count, 1, "one miss keeps with persist 2")
        XCTAssertEqual(tr.tracks[0].misses, 1)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: []))
        XCTAssertEqual(tr.tracks.count, 0, "second miss removes")

        // persist 1 removes on first miss
        let tr1 = DisplayTracker(frameWidth: 320, frameHeight: 200, config: TrackerConfig(persistPasses: 1, minPersistSeconds: 0))
        tr1.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr1.applyDetections(seq: 0, boxes: [b]))
        XCTAssertTrue(tr1.applyDetections(seq: 0, boxes: []))
        XCTAssertEqual(tr1.tracks.count, 0)

        // evicted seq -> false, no change
        var cfg2 = TrackerConfig(persistPasses: 8, minPersistSeconds: 0)
        cfg2.historyCapacity = 4
        let tr2 = DisplayTracker(frameWidth: 320, frameHeight: 200, config: cfg2)
        for i in 0..<6 {
            tr2.ingestFrame(seq: UInt32(i), pts: Double(i) / 60.0, pyramid: pyr(base))
        }
        XCTAssertTrue(tr2.applyDetections(seq: 5, boxes: [b]))
        let countBefore = tr2.tracks.count
        XCTAssertFalse(tr2.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr2.tracks.count, countBefore, "evicted seq must not change state")

        // per-edge grow/shrink (shrinkAlpha 0.25)
        let tr3 = DisplayTracker(frameWidth: 320, frameHeight: 200, config: TrackerConfig(persistPasses: 8, shrinkAlpha: 0.25, minPersistSeconds: 0))
        tr3.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr3.applyDetections(seq: 0, boxes: [b]))
        let shifted = TrackBox(x: b.x + 10, y: b.y, w: b.w, h: b.h)
        XCTAssertTrue(tr3.applyDetections(seq: 0, boxes: [shifted]))
        XCTAssertEqual(tr3.tracks.count, 1, "overlap must match, not create")
        let blended = tr3.tracks[0].box
        XCTAssertEqual(blended.x, 52.5, accuracy: 1e-9)
        XCTAssertEqual(blended.y, b.y, accuracy: 1e-9)
        XCTAssertEqual(blended.w, 67.5, accuracy: 1e-9)
        XCTAssertEqual(blended.h, b.h, accuracy: 1e-9)
    }

    // 7. textureless adopts median
    func testTexturelessAdoptsMedian() {
        let w = 320
        let h = 200
        var base = makeTexture(width: w, height: h, seed: 0x44444)
        // uniform patch 100x100 at (180,80)
        for y in 80..<180 {
            for x in 180..<280 {
                base.pixels[y * w + x] = 128
            }
        }
        let f0 = base
        let f1 = translated(base, dx: 0, dy: 10)
        let tr = DisplayTracker(frameWidth: w, frameHeight: h)
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(f0))
        let t1 = TrackBox(x: 30, y: 30, w: 60, h: 50)
        let t2 = TrackBox(x: 40, y: 120, w: 60, h: 50)
        let flat = TrackBox(x: 200, y: 100, w: 60, h: 50) // inside uniform patch
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [t1, t2, flat]))
        XCTAssertEqual(tr.tracks.count, 3)
        tr.ingestFrame(seq: 1, pts: 1.0 / 60.0, pyramid: pyr(f1))
        XCTAssertEqual(tr.tracks.count, 3)
        // textured tracks should have moved ~10
        // find flat track: it started at (200,100); median shift is ~10 down
        var flatY: Double? = nil
        for t in tr.tracks {
            // identify by x proximity (flat x=200, others x~30-40)
            if abs(t.box.x - 200) < 5 {
                flatY = t.box.y
            }
        }
        XCTAssertNotNil(flatY)
        if let fy = flatY {
            XCTAssertLessThanOrEqual(abs(fy - 110), 3.0, "textureless adopts median shift")
        }
    }

    // 8. fully outside removed
    func testOutsideRemoved() {
        let base = makeTexture(width: 320, height: 200, seed: 0x55555)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200)
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        let b = TrackBox(x: 50, y: 140, w: 60, h: 50) // bottom near 190
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1)
        for i in 1...20 {
            let img = translated(base, dx: 0, dy: i * 10)
            tr.ingestFrame(seq: UInt32(i), pts: Double(i) / 60.0, pyramid: pyr(img))
            if tr.tracks.isEmpty { break }
        }
        XCTAssertEqual(tr.tracks.count, 0, "scrolled outside must be removed")
    }

    // 9. edge coast stays aligned while partially out of frame
    func testEdgeCoastStaysAligned() {
        let base = makeTexture(width: 320, height: 200, seed: 0x55555)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200)
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        let b = TrackBox(x: 50, y: 140, w: 60, h: 50)
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1)
        var sawPartial = false
        var removedAt: Int? = nil
        for i in 1...20 {
            let img = translated(base, dx: 0, dy: i * 10)
            tr.ingestFrame(seq: UInt32(i), pts: Double(i) / 60.0, pyramid: pyr(img))
            if tr.tracks.isEmpty {
                removedAt = i
                break
            }
            let y = tr.tracks[0].box.y
            let expected = 140.0 + Double(i * 10)
            XCTAssertLessThanOrEqual(abs(y - expected), 3.0, "frame \(i) y=\(y) expected \(expected)")
            if y + 50 > 200 { sawPartial = true }
        }
        XCTAssertTrue(sawPartial, "should have observed partially out-of-frame box")
        XCTAssertNotNil(removedAt, "track should eventually leave the frame")
    }

    func testPerfBenchmark() throws {
        guard ProcessInfo.processInfo.environment["BSAFE_PERF"] == "1" else {
            throw XCTSkip("perf benchmark only with BSAFE_PERF=1")
        }
        let w = 2592
        let h = 1676
        let bpr = w * 4 + 64
        var rng: UInt64 = 0x12345678ABCDEF01
        func nextByte() -> UInt8 {
            rng ^= rng &<< 13
            rng ^= rng &>> 7
            rng ^= rng &<< 17
            rng = rng &* 0x2545F4914F6CDD1D &+ 1
            return UInt8(truncatingIfNeeded: (rng >> 33) & 0xFF)
        }
        var bgra = [UInt8](repeating: 0, count: h * bpr)
        for i in 0..<bgra.count { bgra[i] = nextByte() }
        var pyramids = [LumaPyramid]()
        pyramids.reserveCapacity(10)
        let t0 = Date().timeIntervalSinceReferenceDate
        bgra.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for k in 0..<10 {
                let ptr = base.advanced(by: k * bpr)
                pyramids.append(LumaPyramid.fromBGRA(ptr, width: w, height: h, bytesPerRow: bpr))
            }
        }
        let t1 = Date().timeIntervalSinceReferenceDate
        print("PERF fromBGRA avg ms: \((t1 - t0) / 10 * 1000)")
        let bigBox = TrackBox(x: 600, y: 500, w: 400, h: 300)
        let smallBox = TrackBox(x: 600, y: 500, w: 40, h: 30)
        func measure(_ body: () -> Void) -> Double {
            let a = Date().timeIntervalSinceReferenceDate
            body()
            let b = Date().timeIntervalSinceReferenceDate
            return (b - a) * 1000
        }
        var ms12: Double = 0
        for k in 0..<9 {
            ms12 += measure {
                _ = MotionEstimator.estimate(
                    from: pyramids[k], to: pyramids[k + 1],
                    box: bigBox, predicted: Shift.zero, searchRadius: 12)
            }
        }
        print("PERF estimate r=12 400x300 avg ms: \(ms12 / 9)")
        let ms24 = measure {
            _ = MotionEstimator.estimate(
                from: pyramids[0], to: pyramids[1],
                box: bigBox, predicted: Shift.zero, searchRadius: 24)
        }
        print("PERF estimate r=24 400x300 ms: \(ms24)")
        var msSmall: Double = 0
        for k in 0..<9 {
            msSmall += measure {
                _ = MotionEstimator.estimate(
                    from: pyramids[k], to: pyramids[k + 1],
                    box: smallBox, predicted: Shift.zero, searchRadius: 12)
            }
        }
        print("PERF estimate r=12 40x30 avg ms: \(msSmall / 9)")
    }
}
