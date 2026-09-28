import Accelerate
import Foundation

public struct LumaImage {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height, "pixels.count must equal width*height")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    public subscript(x: Int, y: Int) -> UInt8 {
        return pixels[y * width + x]
    }
}

public struct LumaPyramid {
    public let levels: [LumaImage]
    public let downscale: Int

    public var captureWidth: Int { levels[0].width * downscale }
    public var captureHeight: Int { levels[0].height * downscale }

    public init(level0: LumaImage, downscale: Int) {
        precondition(downscale >= 1, "downscale must be >= 1")
        let l1 = LumaPyramid.half(level0)
        let l2 = LumaPyramid.half(l1)
        self.levels = [level0, l1, l2]
        self.downscale = downscale
    }

    static func half(_ img: LumaImage) -> LumaImage {
        let outW = max(1, img.width / 2)
        let outH = max(1, img.height / 2)
        let out = [UInt8](unsafeUninitializedCapacity: outW * outH) { dstBuf, initializedCount in
            guard let dst = dstBuf.baseAddress else {
                initializedCount = 0
                return
            }
            img.pixels.withUnsafeBufferPointer { srcBuf in
                let src = srcBuf.baseAddress!
                let inW = img.width
                let inH = img.height
                for oy in 0..<outH {
                    let y0 = oy * 2
                    let y1 = y0 + 1
                    let hasRow2 = y1 < inH
                    let dstRow = oy * outW
                    if hasRow2 {
                        let row0 = y0 * inW
                        let row1 = y1 * inW
                        for ox in 0..<outW {
                            let x0 = ox * 2
                            if x0 + 1 < inW {
                                let s = UInt32(src[row0 + x0]) + UInt32(src[row0 + x0 + 1])
                                    + UInt32(src[row1 + x0]) + UInt32(src[row1 + x0 + 1])
                                dst[dstRow + ox] = UInt8((s + 2) >> 2)
                            } else {
                                let s = UInt32(src[row0 + x0]) + UInt32(src[row1 + x0])
                                dst[dstRow + ox] = UInt8((s + 1) >> 1)
                            }
                        }
                    } else {
                        let row0 = y0 * inW
                        for ox in 0..<outW {
                            let x0 = ox * 2
                            if x0 + 1 < inW {
                                let s = UInt32(src[row0 + x0]) + UInt32(src[row0 + x0 + 1])
                                dst[dstRow + ox] = UInt8((s + 1) >> 1)
                            } else {
                                dst[dstRow + ox] = src[row0 + x0]
                            }
                        }
                    }
                }
            }
            initializedCount = outW * outH
        }
        return LumaImage(width: outW, height: outH, pixels: out)
    }

    public static func fromBGRA(
        _ base: UnsafeRawPointer,
        width: Int,
        height: Int,
        bytesPerRow: Int,
        maxDimension: Int = 960
    ) -> LumaPyramid {
        let m = width > height ? width : height
        var d = (m + maxDimension - 1) / maxDimension
        if d < 1 { d = 1 }
        let outW = max(1, width / d)
        let outH = max(1, height / d)
        let out: [UInt8] = [UInt8](unsafeUninitializedCapacity: outW * outH) { dstBuf, initializedCount in
            guard let dstBase = dstBuf.baseAddress else {
                initializedCount = 0
                return
            }
            if d == 1 {
                var srcImg = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: base),
                    height: vImagePixelCount(height),
                    width: vImagePixelCount(width),
                    rowBytes: bytesPerRow
                )
                var dstImg = vImage_Buffer(
                    data: UnsafeMutableRawPointer(dstBase),
                    height: vImagePixelCount(height),
                    width: vImagePixelCount(width),
                    rowBytes: outW
                )
                let matrix: [Int16] = [29, 150, 77, 0]
                let err: vImage_Error = matrix.withUnsafeBufferPointer { mPtr in
                    withUnsafePointer(to: &srcImg) { sPtr in
                        withUnsafeMutablePointer(to: &dstImg) { dPtr in
                            vImageMatrixMultiply_ARGB8888ToPlanar8(
                                sPtr, dPtr, mPtr.baseAddress!, 256, nil, 0,
                                vImage_Flags(kvImageNoFlags))
                        }
                    }
                }
                _ = err
            } else {
                let fullCount = width * height
                let fullPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: fullCount)
                defer { fullPtr.deallocate() }
                var srcImg = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: base),
                    height: vImagePixelCount(height),
                    width: vImagePixelCount(width),
                    rowBytes: bytesPerRow
                )
                var tmpImg = vImage_Buffer(
                    data: UnsafeMutableRawPointer(fullPtr),
                    height: vImagePixelCount(height),
                    width: vImagePixelCount(width),
                    rowBytes: width
                )
                let matrix: [Int16] = [29, 150, 77, 0]
                let err: vImage_Error = matrix.withUnsafeBufferPointer { mPtr in
                    withUnsafePointer(to: &srcImg) { sPtr in
                        withUnsafeMutablePointer(to: &tmpImg) { dPtr in
                            vImageMatrixMultiply_ARGB8888ToPlanar8(
                                sPtr, dPtr, mPtr.baseAddress!, 256, nil, 0,
                                vImage_Flags(kvImageNoFlags))
                        }
                    }
                }
                _ = err
                let rowAcc = UnsafeMutablePointer<UInt32>.allocate(capacity: width)
                defer { rowAcc.deallocate() }
                for oy in 0..<outH {
                    let y0 = oy * d
                    var y1 = y0 + d
                    if y1 > height { y1 = height }
                    let hCount = y1 - y0
                    let firstOff = y0 * width
                    for x in 0..<width {
                        rowAcc[x] = UInt32(fullPtr[firstOff + x])
                    }
                    if hCount > 1 {
                        for y in (y0 + 1)..<y1 {
                            let ro = y * width
                            for x in 0..<width {
                                rowAcc[x] += UInt32(fullPtr[ro + x])
                            }
                        }
                    }
                    let dstRow = oy * outW
                    for ox in 0..<outW {
                        let x0 = ox * d
                        var x1 = x0 + d
                        if x1 > width { x1 = width }
                        var s: UInt32 = 0
                        for x in x0..<x1 {
                            s += rowAcc[x]
                        }
                        let cnt = (x1 - x0) * hCount
                        dstBase[dstRow + ox] = UInt8((s + UInt32(cnt / 2)) / UInt32(cnt))
                    }
                }
            }
            initializedCount = outW * outH
        }
        let l0 = LumaImage(width: outW, height: outH, pixels: out)
        return LumaPyramid(level0: l0, downscale: d)
    }
}
