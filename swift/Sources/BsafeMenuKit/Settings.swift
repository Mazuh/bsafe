import Foundation

public enum Censor: String, Codable, CaseIterable {
    case body
    case all
    case female
    case male
}

public enum Style: String, Codable, CaseIterable {
    case pixelate
    case blur
    case black
}

public enum Quality: String, Codable, CaseIterable {
    case fast
    case standard
    case strict
}

public struct BsafeSettings: Codable, Equatable {
    public var censor: Censor
    public var feet: Bool
    public var style: Style
    public var padding: Double
    public var minPadding: Int
    public var quality: Quality

    public init(
        censor: Censor = .body,
        feet: Bool = true,
        style: Style = .pixelate,
        padding: Double = 0.4,
        minPadding: Int = 24,
        quality: Quality = .standard
    ) {
        self.censor = censor
        self.feet = feet
        self.style = style
        self.padding = padding
        self.minPadding = minPadding
        self.quality = quality
    }
}

/// Format a padding fraction minimally, e.g. 0.4 -> "0.4", 0.0 -> "0".
public func formatPadding(_ value: Double) -> String {
    let clamped = min(0.6, max(0.0, value))
    var s = String(format: "%.2f", clamped)
    while s.hasSuffix("0") { s.removeLast() }
    if s.hasSuffix(".") { s.removeLast() }
    return s
}

/// Build the `bsafe start ...` argument list for the given settings.
/// Always starts with ["start", "--stats"]; the feet flag is always explicit.
public func arguments(for settings: BsafeSettings) -> [String] {
    var args = ["start", "--stats"]
    args += ["--censor", settings.censor.rawValue]
    args.append(settings.feet ? "--feet" : "--no-feet")
    args += ["--padding", formatPadding(settings.padding)]
    args += ["--min-padding", "\(min(64, max(0, settings.minPadding)))"]
    switch settings.quality {
    case .fast:
        args += ["--extra-scales", "none", "--detect-scale", "1.0"]
    case .standard:
        args += ["--extra-scales", "0.5", "--detect-scale", "1.0"]
    case .strict:
        args += ["--extra-scales", "0.5", "--detect-scale", "1.5"]
    }
    switch settings.style {
    case .pixelate:
        args += ["--pixels", "1.0", "--blur", "0"]
    case .blur:
        args += ["--blur", "1.0", "--pixels", "0"]
    case .black:
        args += ["--pixels", "0", "--blur", "0"]
    }
    return args
}
