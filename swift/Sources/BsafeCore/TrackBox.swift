import Foundation

public struct TrackBox: Equatable {
    public var x: Double
    public var y: Double
    public var w: Double
    public var h: Double

    public init(x: Double, y: Double, w: Double, h: Double) {
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    }

    public var area: Double { w * h }
    public var centerX: Double { x + w / 2.0 }
    public var centerY: Double { y + h / 2.0 }

    public func iou(_ o: TrackBox) -> Double {
        let ix0 = x > o.x ? x : o.x
        let iy0 = y > o.y ? y : o.y
        let ix1 = (x + w) < (o.x + o.w) ? (x + w) : (o.x + o.w)
        let iy1 = (y + h) < (o.y + o.h) ? (y + h) : (o.y + o.h)
        let iw = ix1 - ix0
        let ih = iy1 - iy0
        if iw <= 0 || ih <= 0 { return 0 }
        let inter = iw * ih
        let u = area + o.area - inter
        if u <= 0 { return 0 }
        return inter / u
    }

    public func union(_ o: TrackBox) -> TrackBox {
        let x0 = x < o.x ? x : o.x
        let y0 = y < o.y ? y : o.y
        let x1 = (x + w) > (o.x + o.w) ? (x + w) : (o.x + o.w)
        let y1 = (y + h) > (o.y + o.h) ? (y + h) : (o.y + o.h)
        return TrackBox(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
    }

    public func offsetBy(dx: Double, dy: Double) -> TrackBox {
        TrackBox(x: x + dx, y: y + dy, w: w, h: h)
    }

    public func contains(x px: Double, y py: Double) -> Bool {
        if w <= 0 || h <= 0 { return false }
        return px >= x && px < x + w && py >= y && py < y + h
    }

    public func clipped(width fw: Double, height fh: Double) -> TrackBox? {
        let x0 = x > 0 ? x : 0
        let y0 = y > 0 ? y : 0
        let x1 = (x + w) < fw ? (x + w) : fw
        let y1 = (y + h) < fh ? (y + h) : fh
        if x1 <= x0 || y1 <= y0 { return nil }
        return TrackBox(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
    }
}

public struct Shift: Equatable {
    public var dx: Double
    public var dy: Double

    public init(dx: Double, dy: Double) {
        self.dx = dx
        self.dy = dy
    }

    public static let zero = Shift(dx: 0, dy: 0)
}
