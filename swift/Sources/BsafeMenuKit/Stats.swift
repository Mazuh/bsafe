import Foundation

public struct LiveStats: Equatable {
    public let rate: Double
    public let detectMs: Double

    public init(rate: Double, detectMs: Double) {
        self.rate = rate
        self.detectMs = detectMs
    }
}

/// Strip ANSI CSI escape sequences (e.g. color codes) from a line.
public func stripANSI(_ line: String) -> String {
    var out = ""
    out.reserveCapacity(line.count)
    var i = line.startIndex
    while i != line.endIndex {
        let ch = line[i]
        if ch == "\u{1B}" {
            let next = line.index(after: i)
            if next != line.endIndex, line[next] == "[" {
                var j = line.index(after: next)
                while j != line.endIndex {
                    let c = line[j]
                    j = line.index(after: j)
                    if c >= "@", c <= "~" { break }
                }
                i = j
            } else {
                i = next
            }
            continue
        }
        out.append(ch)
        i = line.index(after: i)
    }
    return out
}

private func number(after key: String, in line: String) -> Double? {
    guard let r = line.range(of: key) else { return nil }
    let rest = line[r.upperBound...]
    var end = rest.startIndex
    let allowed = Set("0123456789.+-eE")
    while end != rest.endIndex, allowed.contains(rest[end]) {
        end = rest.index(after: end)
    }
    return Double(String(rest[..<end]))
}

/// Parse a `live stats (window): ...` line. Returns nil for any other line
/// (native stats, startup banners, errors, garbage). ANSI-wrapped lines parse.
public func parseStatsLine(_ line: String) -> LiveStats? {
    let clean = stripANSI(line)
    guard clean.contains("live stats") else { return nil }
    guard let rate = number(after: "rate=", in: clean),
          let ms = number(after: "avg_detect_ms=", in: clean)
    else { return nil }
    return LiveStats(rate: rate, detectMs: ms)
}
