// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BsafeCapture",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "BsafeCapture", targets: ["BsafeCapture"]),
        .executable(name: "bsafe-replay", targets: ["BsafeReplay"]),
        .executable(name: "BsafeMenuBar", targets: ["BsafeMenuBar"]),
    ],
    targets: [
        .target(
            name: "BsafeCore",
            path: "Sources/BsafeCore"
        ),
        .executableTarget(
            name: "BsafeCapture",
            dependencies: ["BsafeCore"],
            path: "Sources/BsafeCapture"
        ),
        .executableTarget(
            name: "BsafeReplay",
            dependencies: ["BsafeCore"],
            path: "Sources/BsafeReplay"
        ),
        .target(
            name: "BsafeMenuKit",
            path: "Sources/BsafeMenuKit"
        ),
        .executableTarget(
            name: "BsafeMenuBar",
            dependencies: ["BsafeMenuKit"],
            path: "Sources/BsafeMenuBar"
        ),
        .testTarget(
            name: "BsafeCoreTests",
            dependencies: ["BsafeCore"],
            path: "Tests/BsafeCoreTests"
        ),
        .testTarget(
            name: "BsafeMenuKitTests",
            dependencies: ["BsafeMenuKit"],
            path: "Tests/BsafeMenuKitTests"
        ),
    ]
)
