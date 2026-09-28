import Foundation

public enum MotionEstimator {
    private static let stdThreshold = 2.5
    private static let madAccept = 10.0
    private static let validFraction = 0.75
    private static let coarseMaxSamples = 16
    private static let fineMaxSamples = 24
    private static let changeT = 3
    private static let inT = 8
    /// A sample votes for the hypothesis (shift 0 or B) that explains it at least this much better.
    private static let voteMargin = 4
    private static let minChanged = 8

    public static func estimate(
        from prev: LumaPyramid,
        to cur: LumaPyramid,
        box: TrackBox,
        predicted: Shift,
        searchRadius: Int
    ) -> Shift? {
        let d = prev.downscale
        if prev.levels.count != 3 || cur.levels.count != 3 { return nil }
        if d < 1 { return nil }
        let s0 = d
        let s1 = d * 2
        let s2 = d * 4
        let frameW = Double(min(prev.captureWidth, cur.captureWidth))
        let frameH = Double(min(prev.captureHeight, cur.captureHeight))
        if frameW <= 0 || frameH <= 0 { return nil }

        let c2x = Int((predicted.dx / Double(s2)).rounded())
        let c2y = Int((predicted.dy / Double(s2)).rounded())
        let c1x = Int((predicted.dx / Double(s1)).rounded())
        let c1y = Int((predicted.dy / Double(s1)).rounded())
        let c0x = Int((predicted.dx / Double(s0)).rounded())
        let c0y = Int((predicted.dy / Double(s0)).rounded())

        // ---- L0 grid first: static-target early exit + reuse for L0 refine ----
        var l0Grid: Samples? = nil
        var l0Interior: SampleSet? = nil
        do {
            let region0 = refineCaptureRegion(box: box, sL: s0)
            if let lr0 = intersectCaptureRegion(region0, frameW: frameW, frameH: frameH),
                let lev0 = levelRect(capture: lr0, scale: s0),
                lev0.w >= 4 && lev0.h >= 4,
                let g0 = makeSamples(
                    prev: prev.levels[0], rect: lev0,
                    curWidth: cur.levels[0].width, maxSide: fineMaxSamples),
                g0.std >= stdThreshold
            {
                let interior = interiorSet(from: g0, box: box, s0: s0)
                if interior.count > 0 {
                    let cc = changedCount(interior, cur: cur.levels[0])
                    if cc < max(4, Int(0.10 * Double(interior.count))) {
                        return Shift.zero
                    }
                }
                l0Grid = g0
                l0Interior = interior
            }
        }

        // ---- Coarse L2 ----
        var best2: (x: Int, y: Int)? = nil
        var best2mad: Double = 0
        do {
            let region = coarseCaptureRegion(box: box, s2: s2)
            if let lr = intersectCaptureRegion(region, frameW: frameW, frameH: frameH),
                let lev = levelRect(capture: lr, scale: s2)
            {
                let w2 = lev.w
                let h2 = lev.h
                if w2 >= 4 && h2 >= 4 {
                    let prevL = prev.levels[2]
                    let curL = cur.levels[2]
                    if let samp = makeSamples(
                        prev: prevL, rect: lev, curWidth: curL.width, maxSide: coarseMaxSamples)
                    {
                        if samp.std >= stdThreshold {
                            let full = flatSet(from: samp)
                            let ch = changedSubset(full, cur: curL)
                            let use = ch.count >= minChanged ? ch : full
                            if let found = searchBest(
                                samples: use, cur: curL,
                                centerX: c2x, centerY: c2y, radius: searchRadius,
                                extraZero: true
                            ) {
                                best2 = (found.x, found.y)
                                best2mad = found.mad
                            }
                        }
                    }
                }
            }
        }

        // ---- Refine L1 ----
        var best1: (x: Int, y: Int)? = nil
        var best1mad: Double = 0
        do {
            let region = refineCaptureRegion(box: box, sL: s1)
            if let lr = intersectCaptureRegion(region, frameW: frameW, frameH: frameH),
                let lev = levelRect(capture: lr, scale: s1),
                lev.w >= 4 && lev.h >= 4,
                let samp = makeSamples(
                    prev: prev.levels[1], rect: lev,
                    curWidth: cur.levels[1].width, maxSide: fineMaxSamples),
                samp.std >= stdThreshold
            {
                let full = flatSet(from: samp)
                let ch = changedSubset(full, cur: cur.levels[1])
                let use = ch.count >= minChanged ? ch : full
                let cx: Int
                let cy: Int
                if let b2 = best2 { cx = b2.x * 2; cy = b2.y * 2 }
                else { cx = c1x; cy = c1y }
                if let found = searchBest(
                    samples: use, cur: cur.levels[1],
                    centerX: cx, centerY: cy, radius: 2,
                    extraZero: false
                ) {
                    best1 = (found.x, found.y)
                    best1mad = found.mad
                } else {
                    if let b2 = best2 { best1 = (b2.x * 2, b2.y * 2); best1mad = best2mad }
                    else { best1 = nil }
                }
            } else {
                if let b2 = best2 { best1 = (b2.x * 2, b2.y * 2); best1mad = best2mad }
                else { best1 = nil }
            }
        }

        // ---- Refine L0 ----
        var best0: (x: Int, y: Int)? = nil
        var best0mad: Double = 0
        var l0Use: SampleSet? = nil
        var l0UsedChanged = false
        var l0Searched = false
        do {
            if let samp = l0Grid {
                let full = flatSet(from: samp)
                let ch = changedSubset(full, cur: cur.levels[0])
                let use: SampleSet
                if ch.count >= minChanged {
                    use = ch
                    l0UsedChanged = true
                } else {
                    use = full
                    l0UsedChanged = false
                }
                l0Use = use
                let cx: Int
                let cy: Int
                if let b1 = best1 { cx = b1.x * 2; cy = b1.y * 2 }
                else if let b2 = best2 { cx = b2.x * 4; cy = b2.y * 4 }
                else { cx = c0x; cy = c0y }
                if let found = searchBest(
                    samples: use, cur: cur.levels[0],
                    centerX: cx, centerY: cy, radius: 2,
                    extraZero: false
                ) {
                    best0 = (found.x, found.y)
                    best0mad = found.mad
                    l0Searched = true
                } else {
                    if let b1 = best1 { best0 = (b1.x * 2, b1.y * 2); best0mad = best1mad }
                    else if let b2 = best2 { best0 = (b2.x * 4, b2.y * 4); best0mad = best2mad }
                    else { best0 = nil }
                    l0Searched = false
                }
            } else {
                if let b1 = best1 { best0 = (b1.x * 2, b1.y * 2); best0mad = best1mad }
                else if let b2 = best2 { best0 = (b2.x * 4, b2.y * 4); best0mad = best2mad }
                else { best0 = nil }
                l0Searched = false
            }
        }

        guard let b0 = best0 else { return nil }

        if !l0Searched {
            if best0mad > madAccept { return nil }
            return Shift(dx: Double(b0.x) * Double(s0), dy: Double(b0.y) * Double(s0))
        }

        guard let use0 = l0Use else { return nil }

        if b0.x == 0 && b0.y == 0 { return Shift.zero }

        if let interior = l0Interior, interior.count > 0 {
            var exB = 0
            var exZ = 0
            let curL0 = cur.levels[0]
            curL0.pixels.withUnsafeBufferPointer { curBuf in
                interior.xs.withUnsafeBufferPointer { xsBuf in
                    interior.ys.withUnsafeBufferPointer { ysBuf in
                        interior.vals.withUnsafeBufferPointer { valsBuf in
                            guard let curBase = curBuf.baseAddress,
                                let xsBase = xsBuf.baseAddress,
                                let ysBase = ysBuf.baseAddress,
                                let valsBase = valsBuf.baseAddress
                            else { return }
                            let cw = curL0.width
                            let ch = curL0.height
                            let n = valsBuf.count
                            for k in 0..<n {
                                let x = xsBase[k]
                                let y = ysBase[k]
                                if x < 0 || x >= cw || y < 0 || y >= ch { continue }
                                let sx = x + b0.x
                                let sy = y + b0.y
                                if sx < 0 || sx >= cw || sy < 0 || sy >= ch { continue }
                                let v = valsBase[k]
                                let c0v = curBase[y * cw + x]
                                let cbv = curBase[sy * cw + sx]
                                let d0: Int = v >= c0v ? Int(v - c0v) : Int(c0v - v)
                                let dB: Int = v >= cbv ? Int(v - cbv) : Int(cbv - v)
                                if dB + voteMargin <= d0 {
                                    exB += 1
                                } else if d0 + voteMargin <= dB {
                                    exZ += 1
                                }
                            }
                        }
                    }
                }
            }
            if exB <= exZ { return Shift.zero }
        }

        var deltaX = 0.0
        var deltaY = 0.0
        let curL0 = cur.levels[0]
        let m0 = best0mad
        if let mxm = madAt(samples: use0, cur: curL0, ox: b0.x - 1, oy: b0.y),
            let mxp = madAt(samples: use0, cur: curL0, ox: b0.x + 1, oy: b0.y)
        {
            let den = 2.0 * (mxm - 2.0 * m0 + mxp)
            if den > 1e-9 {
                var dd = (mxm - mxp) / den
                if dd < -0.5 { dd = -0.5 }
                if dd > 0.5 { dd = 0.5 }
                deltaX = dd
            }
        }
        if let mym = madAt(samples: use0, cur: curL0, ox: b0.x, oy: b0.y - 1),
            let myp = madAt(samples: use0, cur: curL0, ox: b0.x, oy: b0.y + 1)
        {
            let den = 2.0 * (mym - 2.0 * m0 + myp)
            if den > 1e-9 {
                var dd = (mym - myp) / den
                if dd < -0.5 { dd = -0.5 }
                if dd > 0.5 { dd = 0.5 }
                deltaY = dd
            }
        }

        if l0UsedChanged {
            var match = 0
            let total = use0.count
            curL0.pixels.withUnsafeBufferPointer { curBuf in
                use0.xs.withUnsafeBufferPointer { xsBuf in
                    use0.ys.withUnsafeBufferPointer { ysBuf in
                        use0.vals.withUnsafeBufferPointer { valsBuf in
                            guard let curBase = curBuf.baseAddress,
                                let xsBase = xsBuf.baseAddress,
                                let ysBase = ysBuf.baseAddress,
                                let valsBase = valsBuf.baseAddress
                            else { return }
                            let cw = curL0.width
                            let ch = curL0.height
                            let n = valsBuf.count
                            for k in 0..<n {
                                let sx = xsBase[k] + b0.x
                                let sy = ysBase[k] + b0.y
                                if sx < 0 || sx >= cw || sy < 0 || sy >= ch { continue }
                                let v = valsBase[k]
                                let c = curBase[sy * cw + sx]
                                let dd: Int = v >= c ? Int(v - c) : Int(c - v)
                                if dd <= inT { match += 1 }
                            }
                        }
                    }
                }
            }
            if match * 2 < total { return nil }
        } else {
            if m0 > madAccept { return nil }
        }

        return Shift(dx: (Double(b0.x) + deltaX) * Double(s0), dy: (Double(b0.y) + deltaY) * Double(s0))
    }

    // MARK: - Regions

    private static func coarseCaptureRegion(box: TrackBox, s2: Int) -> TrackBox {
        let e = 0.15 * max(box.w, box.h)
        let halfNeed = Double(8 * s2)
        let halfW = max(box.w / 2.0, halfNeed) + e
        let halfH = max(box.h / 2.0, halfNeed) + e
        let cx = box.x + box.w / 2.0
        let cy = box.y + box.h / 2.0
        return TrackBox(x: cx - halfW, y: cy - halfH, w: halfW * 2.0, h: halfH * 2.0)
    }

    private static func refineCaptureRegion(box: TrackBox, sL: Int) -> TrackBox {
        let e = 0.15 * max(box.w, box.h)
        let halfNeed = Double(4 * sL)
        var halfW = box.w / 2.0 + e
        if halfW < halfNeed { halfW = halfNeed }
        var halfH = box.h / 2.0 + e
        if halfH < halfNeed { halfH = halfNeed }
        let cx = box.x + box.w / 2.0
        let cy = box.y + box.h / 2.0
        return TrackBox(x: cx - halfW, y: cy - halfH, w: halfW * 2.0, h: halfH * 2.0)
    }

    private struct CaptureRect {
        var x0: Double
        var y0: Double
        var x1: Double
        var y1: Double
    }

    private static func intersectCaptureRegion(_ r: TrackBox, frameW: Double, frameH: Double) -> CaptureRect? {
        var x0 = r.x
        var y0 = r.y
        var x1 = r.x + r.w
        var y1 = r.y + r.h
        if x0 < 0 { x0 = 0 }
        if y0 < 0 { y0 = 0 }
        if x1 > frameW { x1 = frameW }
        if y1 > frameH { y1 = frameH }
        if x1 <= x0 || y1 <= y0 { return nil }
        return CaptureRect(x0: x0, y0: y0, x1: x1, y1: y1)
    }

    private struct LevelRect {
        var x: Int
        var y: Int
        var w: Int
        var h: Int
    }

    private static func levelRect(capture: CaptureRect, scale: Int) -> LevelRect? {
        let s = Double(scale)
        let lx = Int(floor(capture.x0 / s))
        let ly = Int(floor(capture.y0 / s))
        let x1 = Int(ceil(capture.x1 / s))
        let y1 = Int(ceil(capture.y1 / s))
        let w = x1 - lx
        let h = y1 - ly
        if w <= 0 || h <= 0 { return nil }
        return LevelRect(x: lx, y: ly, w: w, h: h)
    }

    // MARK: - Sampling

    private struct Samples {
        var xs: [Int]
        var ys: [Int]
        var rowOffs: [Int]
        var vals: [UInt8]
        var std: Double
    }

    private struct SampleSet {
        var offs: [Int]
        var xs: [Int]
        var ys: [Int]
        var vals: [UInt8]
        var xMin: Int
        var xMax: Int
        var yMin: Int
        var yMax: Int
        var count: Int { vals.count }
    }

    private static func makeSamples(prev: LumaImage, rect: LevelRect, curWidth: Int, maxSide: Int) -> Samples? {
        var lx = rect.x
        var ly = rect.y
        var w = rect.w
        var h = rect.h
        if lx < 0 { w += lx; lx = 0 }
        if ly < 0 { h += ly; ly = 0 }
        if lx + w > prev.width { w = prev.width - lx }
        if ly + h > prev.height { h = prev.height - ly }
        if w <= 0 || h <= 0 { return nil }
        let limit = maxSide > 0 ? maxSide : fineMaxSamples
        var xs = [Int]()
        var ys = [Int]()
        if w <= limit {
            xs.reserveCapacity(w)
            for i in 0..<w { xs.append(lx + i) }
        } else {
            xs.reserveCapacity(limit)
            for i in 0..<limit { xs.append(lx + (i * w) / limit) }
        }
        if h <= limit {
            ys.reserveCapacity(h)
            for i in 0..<h { ys.append(ly + i) }
        } else {
            ys.reserveCapacity(limit)
            for i in 0..<limit { ys.append(ly + (i * h) / limit) }
        }
        let n = xs.count * ys.count
        if n == 0 { return nil }
        var vals = [UInt8](repeating: 0, count: n)
        prev.pixels.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            var k = 0
            for yy in ys {
                let rowOff = yy * prev.width
                for xx in xs {
                    vals[k] = base[rowOff + xx]
                    k += 1
                }
            }
        }
        var sum = 0.0
        var sumSq = 0.0
        for v in vals {
            let dd = Double(v)
            sum += dd
            sumSq += dd * dd
        }
        let nn = Double(n)
        let mean = sum / nn
        var variance = sumSq / nn - mean * mean
        if variance < 0 { variance = 0 }
        let std = variance.squareRoot()
        let rowOffs = ys.map { $0 * curWidth }
        return Samples(xs: xs, ys: ys, rowOffs: rowOffs, vals: vals, std: std)
    }

    private static func flatSet(from s: Samples) -> SampleSet {
        let nx = s.xs.count
        let ny = s.ys.count
        let n = s.vals.count
        if n == 0 || nx == 0 || ny == 0 {
            return SampleSet(offs: [], xs: [], ys: [], vals: [], xMin: 0, xMax: 0, yMin: 0, yMax: 0)
        }
        var offs = [Int](repeating: 0, count: n)
        var fxs = [Int](repeating: 0, count: n)
        var fys = [Int](repeating: 0, count: n)
        var k = 0
        for iy in 0..<ny {
            let y = s.ys[iy]
            let ro = s.rowOffs[iy]
            for ix in 0..<nx {
                let x = s.xs[ix]
                fxs[k] = x
                fys[k] = y
                offs[k] = ro + x
                k += 1
            }
        }
        return SampleSet(
            offs: offs, xs: fxs, ys: fys, vals: s.vals,
            xMin: s.xs[0], xMax: s.xs[nx - 1], yMin: s.ys[0], yMax: s.ys[ny - 1])
    }

    private static func interiorSet(from s: Samples, box: TrackBox, s0: Int) -> SampleSet {
        let nx = s.xs.count
        let ny = s.ys.count
        let n = s.vals.count
        if n == 0 || nx == 0 || ny == 0 {
            return SampleSet(offs: [], xs: [], ys: [], vals: [], xMin: 0, xMax: 0, yMin: 0, yMax: 0)
        }
        var offs = [Int]()
        var fxs = [Int]()
        var fys = [Int]()
        var fvals = [UInt8]()
        offs.reserveCapacity(n)
        fxs.reserveCapacity(n)
        fys.reserveCapacity(n)
        fvals.reserveCapacity(n)
        var xMin = Int.max
        var xMax = Int.min
        var yMin = Int.max
        var yMax = Int.min
        let bx1 = box.x + box.w
        let by1 = box.y + box.h
        var k = 0
        for iy in 0..<ny {
            let y = s.ys[iy]
            let ro = s.rowOffs[iy]
            let capY = Double(y * s0)
            let insideY = capY >= box.y && capY < by1
            for ix in 0..<nx {
                let x = s.xs[ix]
                let v = s.vals[k]
                k += 1
                if !insideY { continue }
                let capX = Double(x * s0)
                if capX >= box.x && capX < bx1 {
                    offs.append(ro + x)
                    fxs.append(x)
                    fys.append(y)
                    fvals.append(v)
                    if x < xMin { xMin = x }
                    if x > xMax { xMax = x }
                    if y < yMin { yMin = y }
                    if y > yMax { yMax = y }
                }
            }
        }
        if fvals.isEmpty {
            return SampleSet(offs: [], xs: [], ys: [], vals: [], xMin: 0, xMax: 0, yMin: 0, yMax: 0)
        }
        return SampleSet(offs: offs, xs: fxs, ys: fys, vals: fvals, xMin: xMin, xMax: xMax, yMin: yMin, yMax: yMax)
    }

    private static func changedCount(_ s: SampleSet, cur: LumaImage) -> Int {
        let n = s.vals.count
        if n == 0 { return 0 }
        let cw = cur.width
        let ch = cur.height
        let pix = cur.pixels
        let pixCount = pix.count
        var c = 0
        for k in 0..<n {
            let x = s.xs[k]
            let y = s.ys[k]
            if x < 0 || x >= cw || y < 0 || y >= ch { continue }
            let o = s.offs[k]
            if o < 0 || o >= pixCount { continue }
            let a = s.vals[k]
            let b = pix[o]
            let dd: Int = a >= b ? Int(a - b) : Int(b - a)
            if dd > changeT { c += 1 }
        }
        return c
    }

    private static func changedSubset(_ s: SampleSet, cur: LumaImage) -> SampleSet {
        let n = s.vals.count
        if n == 0 {
            return SampleSet(offs: [], xs: [], ys: [], vals: [], xMin: 0, xMax: 0, yMin: 0, yMax: 0)
        }
        let cw = cur.width
        let ch = cur.height
        let pix = cur.pixels
        let pixCount = pix.count
        var offs = [Int]()
        var xs = [Int]()
        var ys = [Int]()
        var vals = [UInt8]()
        offs.reserveCapacity(n)
        xs.reserveCapacity(n)
        ys.reserveCapacity(n)
        vals.reserveCapacity(n)
        var xMin = Int.max
        var xMax = Int.min
        var yMin = Int.max
        var yMax = Int.min
        for k in 0..<n {
            let x = s.xs[k]
            let y = s.ys[k]
            if x < 0 || x >= cw || y < 0 || y >= ch { continue }
            let o = s.offs[k]
            if o < 0 || o >= pixCount { continue }
            let a = s.vals[k]
            let b = pix[o]
            let dd: Int = a >= b ? Int(a - b) : Int(b - a)
            if dd > changeT {
                offs.append(o)
                xs.append(x)
                ys.append(y)
                vals.append(a)
                if x < xMin { xMin = x }
                if x > xMax { xMax = x }
                if y < yMin { yMin = y }
                if y > yMax { yMax = y }
            }
        }
        if vals.isEmpty {
            return SampleSet(offs: [], xs: [], ys: [], vals: [], xMin: 0, xMax: 0, yMin: 0, yMax: 0)
        }
        return SampleSet(offs: offs, xs: xs, ys: ys, vals: vals, xMin: xMin, xMax: xMax, yMin: yMin, yMax: yMax)
    }

    private static func madAt(samples: SampleSet, cur: LumaImage, ox: Int, oy: Int) -> Double? {
        let total = samples.vals.count
        if total == 0 { return nil }
        var result: Double? = nil
        cur.pixels.withUnsafeBufferPointer { curBuf in
            samples.offs.withUnsafeBufferPointer { offsBuf in
                samples.xs.withUnsafeBufferPointer { xsBuf in
                    samples.ys.withUnsafeBufferPointer { ysBuf in
                        samples.vals.withUnsafeBufferPointer { valsBuf in
                            guard let curBase = curBuf.baseAddress,
                                let offsBase = offsBuf.baseAddress,
                                let xsBase = xsBuf.baseAddress,
                                let ysBase = ysBuf.baseAddress,
                                let valsBase = valsBuf.baseAddress
                            else { return }
                            let cw = cur.width
                            let ch = cur.height
                            let n = total
                            let xMin = samples.xMin
                            let xMax = samples.xMax
                            let yMin = samples.yMin
                            let yMax = samples.yMax
                            if xMin + ox >= 0 && xMax + ox < cw && yMin + oy >= 0 && yMax + oy < ch {
                                var sum: UInt64 = 0
                                let candBase = oy * cw + ox
                                for k in 0..<n {
                                    let a = UInt32(valsBase[k])
                                    let b = UInt32(curBase[offsBase[k] + candBase])
                                    sum &+= UInt64(a >= b ? (a - b) : (b - a))
                                }
                                result = Double(sum) / Double(total)
                            } else {
                                var sum: UInt64 = 0
                                var valid = 0
                                for k in 0..<n {
                                    let sx = xsBase[k] + ox
                                    let sy = ysBase[k] + oy
                                    if sy >= 0 && sy < ch && sx >= 0 && sx < cw {
                                        let a = UInt32(valsBase[k])
                                        let b = UInt32(curBase[sy * cw + sx])
                                        sum &+= UInt64(a >= b ? (a - b) : (b - a))
                                        valid += 1
                                    }
                                }
                                if valid > 0 && Double(valid) >= validFraction * Double(total) {
                                    result = Double(sum) / Double(valid)
                                } else {
                                    result = nil
                                }
                            }
                        }
                    }
                }
            }
        }
        return result
    }

    @inline(__always)
    private static func considerCandidate(
        ox: Int, oy: Int,
        offsBase: UnsafePointer<Int>,
        xsBase: UnsafePointer<Int>,
        ysBase: UnsafePointer<Int>,
        valsBase: UnsafePointer<UInt8>,
        n: Int,
        curBase: UnsafePointer<UInt8>,
        cw: Int, ch: Int,
        xMin: Int, xMax: Int, yMin: Int, yMax: Int,
        total: Int,
        centerX: Int, centerY: Int,
        bestX: inout Int, bestY: inout Int,
        bestMad: inout Double, bestSum: inout UInt64,
        bestIsFull: inout Bool, found: inout Bool
    ) {
        let fast = xMin + ox >= 0 && xMax + ox < cw && yMin + oy >= 0 && yMax + oy < ch
        if fast {
            let candBase = oy * cw + ox
            var sum: UInt64 = 0
            var pruned = false
            if found {
                if bestIsFull {
                    let thresh = bestSum
                    outer: for k in 0..<n {
                        let a = UInt32(valsBase[k])
                        let b = UInt32(curBase[offsBase[k] + candBase])
                        sum &+= UInt64(a >= b ? (a - b) : (b - a))
                        if sum > thresh {
                            pruned = true
                            break outer
                        }
                    }
                } else {
                    let thresh = bestMad * Double(total) + 1e-9
                    outer: for k in 0..<n {
                        let a = UInt32(valsBase[k])
                        let b = UInt32(curBase[offsBase[k] + candBase])
                        sum &+= UInt64(a >= b ? (a - b) : (b - a))
                        if Double(sum) > thresh {
                            pruned = true
                            break outer
                        }
                    }
                }
            } else {
                for k in 0..<n {
                    let a = UInt32(valsBase[k])
                    let b = UInt32(curBase[offsBase[k] + candBase])
                    sum &+= UInt64(a >= b ? (a - b) : (b - a))
                }
            }
            if pruned { return }
            let mad = Double(sum) / Double(total)
            if !found || mad < bestMad - 1e-12 {
                found = true
                bestMad = mad
                bestX = ox
                bestY = oy
                bestSum = sum
                bestIsFull = true
            } else if abs(mad - bestMad) <= 1e-12 {
                let dxOld = bestX - centerX
                let dyOld = bestY - centerY
                let dxNew = ox - centerX
                let dyNew = oy - centerY
                let distOld = dxOld * dxOld + dyOld * dyOld
                let distNew = dxNew * dxNew + dyNew * dyNew
                if distNew < distOld {
                    bestX = ox
                    bestY = oy
                    bestSum = sum
                    bestIsFull = true
                } else if distNew == distOld {
                    let magOld = bestX * bestX + bestY * bestY
                    let magNew = ox * ox + oy * oy
                    if magNew < magOld {
                        bestX = ox
                        bestY = oy
                        bestSum = sum
                        bestIsFull = true
                    }
                }
            }
        } else {
            var sum: UInt64 = 0
            var valid = 0
            for k in 0..<n {
                let sx = xsBase[k] + ox
                let sy = ysBase[k] + oy
                if sy >= 0 && sy < ch && sx >= 0 && sx < cw {
                    let a = UInt32(valsBase[k])
                    let b = UInt32(curBase[sy * cw + sx])
                    sum &+= UInt64(a >= b ? (a - b) : (b - a))
                    valid += 1
                }
            }
            if valid == 0 || Double(valid) < validFraction * Double(total) { return }
            let mad = Double(sum) / Double(valid)
            if !found || mad < bestMad - 1e-12 {
                found = true
                bestMad = mad
                bestX = ox
                bestY = oy
                bestIsFull = false
            } else if abs(mad - bestMad) <= 1e-12 {
                let dxOld = bestX - centerX
                let dyOld = bestY - centerY
                let dxNew = ox - centerX
                let dyNew = oy - centerY
                let distOld = dxOld * dxOld + dyOld * dyOld
                let distNew = dxNew * dxNew + dyNew * dyNew
                if distNew < distOld {
                    bestX = ox
                    bestY = oy
                    bestIsFull = false
                } else if distNew == distOld {
                    let magOld = bestX * bestX + bestY * bestY
                    let magNew = ox * ox + oy * oy
                    if magNew < magOld {
                        bestX = ox
                        bestY = oy
                        bestIsFull = false
                    }
                }
            }
        }
    }

    private static func searchBest(
        samples: SampleSet,
        cur: LumaImage,
        centerX: Int,
        centerY: Int,
        radius: Int,
        extraZero: Bool
    ) -> (x: Int, y: Int, mad: Double)? {
        var bestX = 0
        var bestY = 0
        var bestMad = Double.infinity
        var bestSum: UInt64 = 0
        var bestIsFull = false
        var found = false
        let total = samples.vals.count
        if total == 0 { return nil }
        cur.pixels.withUnsafeBufferPointer { curBuf in
            samples.offs.withUnsafeBufferPointer { offsBuf in
                samples.xs.withUnsafeBufferPointer { xsBuf in
                    samples.ys.withUnsafeBufferPointer { ysBuf in
                        samples.vals.withUnsafeBufferPointer { valsBuf in
                            guard let curBase = curBuf.baseAddress,
                                let offsBase = offsBuf.baseAddress,
                                let xsBase = xsBuf.baseAddress,
                                let ysBase = ysBuf.baseAddress,
                                let valsBase = valsBuf.baseAddress
                            else { return }
                            let cw = cur.width
                            let ch = cur.height
                            let n = total
                            let xMin = samples.xMin
                            let xMax = samples.xMax
                            let yMin = samples.yMin
                            let yMax = samples.yMax
                            for oy in (centerY - radius)...(centerY + radius) {
                                for ox in (centerX - radius)...(centerX + radius) {
                                    considerCandidate(
                                        ox: ox, oy: oy,
                                        offsBase: offsBase,
                                        xsBase: xsBase,
                                        ysBase: ysBase,
                                        valsBase: valsBase,
                                        n: n,
                                        curBase: curBase,
                                        cw: cw, ch: ch,
                                        xMin: xMin, xMax: xMax, yMin: yMin, yMax: yMax,
                                        total: total,
                                        centerX: centerX, centerY: centerY,
                                        bestX: &bestX, bestY: &bestY,
                                        bestMad: &bestMad, bestSum: &bestSum,
                                        bestIsFull: &bestIsFull, found: &found
                                    )
                                }
                            }
                            if extraZero {
                                let zeroInX = (centerX - radius) <= 0 && 0 <= (centerX + radius)
                                let zeroInY = (centerY - radius) <= 0 && 0 <= (centerY + radius)
                                if !(zeroInX && zeroInY) {
                                    considerCandidate(
                                        ox: 0, oy: 0,
                                        offsBase: offsBase,
                                        xsBase: xsBase,
                                        ysBase: ysBase,
                                        valsBase: valsBase,
                                        n: n,
                                        curBase: curBase,
                                        cw: cw, ch: ch,
                                        xMin: xMin, xMax: xMax, yMin: yMin, yMax: yMax,
                                        total: total,
                                        centerX: centerX, centerY: centerY,
                                        bestX: &bestX, bestY: &bestY,
                                        bestMad: &bestMad, bestSum: &bestSum,
                                        bestIsFull: &bestIsFull, found: &found
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        if !found { return nil }
        return (bestX, bestY, bestMad)
    }
}
