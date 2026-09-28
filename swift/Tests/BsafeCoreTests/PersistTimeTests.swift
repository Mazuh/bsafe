import Foundation
import XCTest
@testable import BsafeCore

final class PersistTimeTests: XCTestCase {
    let b = TrackBox(x: 50, y: 50, w: 60, h: 50)

    func testTimeFloorKeepsThenRemoves() {
        var cfg = TrackerConfig(persistPasses: 2, minPersistSeconds: 0.6)
        cfg.historyCapacity = 16
        let base = makeTexture(width: 320, height: 200, seed: 0x6A11)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200, config: cfg)
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        XCTAssertEqual(tr.tracks.count, 1)
        let ptsList: [Double] = [0.1, 0.2, 0.3]
        for (k, pts) in ptsList.enumerated() {
            tr.ingestFrame(seq: UInt32(k + 1), pts: pts, pyramid: pyr(base))
            XCTAssertTrue(tr.applyDetections(seq: UInt32(k + 1), boxes: []))
            XCTAssertEqual(tr.tracks.count, 1, "pts=\(pts) must keep")
        }
        XCTAssertEqual(tr.tracks[0].misses, 3)
        tr.ingestFrame(seq: 4, pts: 0.7, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 4, boxes: []))
        XCTAssertEqual(tr.tracks.count, 0, "pts=0.7 must remove")
    }

    func testPassCountStillGatesWhenTimeSatisfied() {
        var cfg = TrackerConfig(persistPasses: 8, minPersistSeconds: 0.1)
        cfg.historyCapacity = 32
        let base = makeTexture(width: 320, height: 200, seed: 0x6A12)
        let tr = DisplayTracker(frameWidth: 320, frameHeight: 200, config: cfg)
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        for i in 1...7 {
            tr.ingestFrame(seq: UInt32(i), pts: Double(i) / 60.0, pyramid: pyr(base))
            XCTAssertTrue(tr.applyDetections(seq: UInt32(i), boxes: []))
            XCTAssertEqual(tr.tracks.count, 1, "after \(i) misses must keep")
        }
        // Time gate already satisfied (7/60 ~= 0.117 >= 0.1) but pass count gates.
        XCTAssertEqual(tr.tracks[0].misses, 7)
        tr.ingestFrame(seq: 8, pts: 8.0 / 60.0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 8, boxes: []))
        XCTAssertEqual(tr.tracks.count, 0, "8th miss must remove")
    }

    func testShrinkTimeNormalizedThirdSecond() {
        let base = makeTexture(width: 320, height: 200, seed: 0x6A13)
        let tr = DisplayTracker(
            frameWidth: 320, frameHeight: 200,
            config: TrackerConfig(persistPasses: 8, shrinkAlpha: 0.25, minPersistSeconds: 0))
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
        tr.ingestFrame(seq: 1, pts: 1.0 / 30.0, pyramid: pyr(base))
        let narrow = TrackBox(x: 60, y: 50, w: 50, h: 50)
        XCTAssertTrue(tr.applyDetections(seq: 1, boxes: [narrow]))
        XCTAssertEqual(tr.tracks.count, 1)
        let box = tr.tracks[0].box
        let kEff = 1.0 - pow(0.75, (1.0 / 30.0) / 0.1)
        XCTAssertEqual(box.x, 50.0 + 10.0 * kEff, accuracy: 1e-9, "x=\(box.x)")
        XCTAssertLessThanOrEqual(abs(box.x - 50.914), 0.01, "x=\(box.x) expected ~50.914")
        XCTAssertGreaterThan(abs(box.x - 52.5), 1.0, "must not use legacy per-pass 52.5")
    }

    func testShrinkTimeNormalizedTenthAndCappedGap() {
        let base = makeTexture(width: 320, height: 200, seed: 0x6A14)
        // dt = 0.1 -> legacy value 52.5
        let tr1 = DisplayTracker(
            frameWidth: 320, frameHeight: 200,
            config: TrackerConfig(persistPasses: 8, shrinkAlpha: 0.25, minPersistSeconds: 0))
        tr1.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr1.applyDetections(seq: 0, boxes: [b]))
        tr1.ingestFrame(seq: 1, pts: 0.1, pyramid: pyr(base))
        XCTAssertTrue(tr1.applyDetections(seq: 1, boxes: [TrackBox(x: 60, y: 50, w: 50, h: 50)]))
        XCTAssertEqual(tr1.tracks[0].box.x, 52.5, accuracy: 1e-9)
        // A match after a 1 s gap shrinks by at most one 0.1 s step -> still 52.5, not a collapse to 60.
        let tr2 = DisplayTracker(
            frameWidth: 320, frameHeight: 200,
            config: TrackerConfig(persistPasses: 8, shrinkAlpha: 0.25, minPersistSeconds: 0))
        tr2.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr2.applyDetections(seq: 0, boxes: [b]))
        tr2.ingestFrame(seq: 1, pts: 1.0, pyramid: pyr(base))
        XCTAssertTrue(tr2.applyDetections(seq: 1, boxes: [TrackBox(x: 60, y: 50, w: 50, h: 50)]))
        XCTAssertEqual(tr2.tracks[0].box.x, 52.5, accuracy: 1e-9)
    }

    func testLateSeqNeitherShrinksNorRewindsClock() {
        let base = makeTexture(width: 320, height: 200, seed: 0x6A15)
        let tr = DisplayTracker(
            frameWidth: 320, frameHeight: 200,
            config: TrackerConfig(persistPasses: 1, shrinkAlpha: 0.25, minPersistSeconds: 0.6))
        tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        tr.ingestFrame(seq: 1, pts: 0.5, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 1, boxes: [b]))
        // Late result for the older frame: matches, must not shrink or move lastMatchPts back to 0.
        XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [TrackBox(x: 60, y: 50, w: 50, h: 50)]))
        XCTAssertEqual(tr.tracks.count, 1)
        XCTAssertEqual(tr.tracks[0].box.x, 50, accuracy: 1e-9)
        XCTAssertEqual(tr.tracks[0].box.w, 60, accuracy: 1e-9)
        // Miss at pts 0.9: only 0.4 s since the real last match (0.5), so the track stays.
        tr.ingestFrame(seq: 2, pts: 0.9, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 2, boxes: []))
        XCTAssertEqual(tr.tracks.count, 1, "clock must not rewind to the late frame's pts")
        tr.ingestFrame(seq: 3, pts: 1.2, pyramid: pyr(base))
        XCTAssertTrue(tr.applyDetections(seq: 3, boxes: []))
        XCTAssertEqual(tr.tracks.count, 0)
    }

    func testOutwardSnapIgnoresDt() {
        let base = makeTexture(width: 320, height: 200, seed: 0x6A15)
        let grown = TrackBox(x: 40, y: 45, w: 80, h: 60)
        for dt in [1.0 / 30.0, 0.2] {
            let tr = DisplayTracker(
                frameWidth: 320, frameHeight: 200,
                config: TrackerConfig(persistPasses: 8, shrinkAlpha: 0.25, minPersistSeconds: 0))
            tr.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
            XCTAssertTrue(tr.applyDetections(seq: 0, boxes: [b]))
            tr.ingestFrame(seq: 1, pts: dt, pyramid: pyr(base))
            XCTAssertTrue(tr.applyDetections(seq: 1, boxes: [grown]))
            XCTAssertEqual(tr.tracks.count, 1, "dt=\(dt)")
            let box = tr.tracks[0].box
            XCTAssertEqual(box.x, 40, accuracy: 1e-9, "dt=\(dt) x=\(box.x)")
            XCTAssertEqual(box.y, 45, accuracy: 1e-9, "dt=\(dt) y=\(box.y)")
            XCTAssertEqual(box.w, 80, accuracy: 1e-9, "dt=\(dt) w=\(box.w)")
            XCTAssertEqual(box.h, 60, accuracy: 1e-9, "dt=\(dt) h=\(box.h)")
        }
    }

    func testMinPersistSanitization() {
        XCTAssertEqual(TrackerConfig(minPersistSeconds: .nan).minPersistSeconds, 1.0, accuracy: 1e-9)
        XCTAssertEqual(TrackerConfig(minPersistSeconds: -1).minPersistSeconds, 0, accuracy: 1e-9)
    }

    func testShrinkAlphaZeroOneWithDt() {
        let base = makeTexture(width: 320, height: 200, seed: 0x6A16)
        let narrow = TrackBox(x: 60, y: 50, w: 50, h: 50)
        // alpha 0 never shrinks even with dt 0.2
        let tr0 = DisplayTracker(
            frameWidth: 320, frameHeight: 200,
            config: TrackerConfig(persistPasses: 8, shrinkAlpha: 0, minPersistSeconds: 0))
        tr0.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr0.applyDetections(seq: 0, boxes: [b]))
        tr0.ingestFrame(seq: 1, pts: 0.2, pyramid: pyr(base))
        XCTAssertTrue(tr0.applyDetections(seq: 1, boxes: [narrow]))
        XCTAssertEqual(tr0.tracks[0].box.x, 50, accuracy: 1e-9)
        XCTAssertEqual(tr0.tracks[0].box.w, 60, accuracy: 1e-9)
        // alpha 1 snaps even with small dt
        let tr1 = DisplayTracker(
            frameWidth: 320, frameHeight: 200,
            config: TrackerConfig(persistPasses: 8, shrinkAlpha: 1.0, minPersistSeconds: 0))
        tr1.ingestFrame(seq: 0, pts: 0, pyramid: pyr(base))
        XCTAssertTrue(tr1.applyDetections(seq: 0, boxes: [b]))
        tr1.ingestFrame(seq: 1, pts: 1.0 / 60.0, pyramid: pyr(base))
        XCTAssertTrue(tr1.applyDetections(seq: 1, boxes: [narrow]))
        XCTAssertEqual(tr1.tracks[0].box.x, 60, accuracy: 1e-9)
        XCTAssertEqual(tr1.tracks[0].box.w, 50, accuracy: 1e-9)
    }
}
