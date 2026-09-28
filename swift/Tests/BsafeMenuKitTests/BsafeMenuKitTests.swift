import Darwin
import XCTest
@testable import BsafeMenuKit

final class BsafeMenuKitTests: XCTestCase {
    // MARK: - arguments()

    func testDefaultArguments() {
        XCTAssertEqual(
            arguments(for: BsafeSettings()),
            ["start", "--stats", "--censor", "body", "--feet",
             "--padding", "0.4", "--min-padding", "24",
             "--extra-scales", "0.5", "--detect-scale", "1.0",
             "--pixels", "1.0", "--blur", "0"]
        )
    }

    func testStyleArguments() {
        var s = BsafeSettings()
        s.style = .blur
        XCTAssertEqual(
            arguments(for: s),
            ["start", "--stats", "--censor", "body", "--feet",
             "--padding", "0.4", "--min-padding", "24",
             "--extra-scales", "0.5", "--detect-scale", "1.0",
             "--blur", "1.0", "--pixels", "0"]
        )
        s.style = .black
        XCTAssertEqual(
            Array(arguments(for: s).suffix(4)),
            ["--pixels", "0", "--blur", "0"]
        )
    }

    func testQualityArguments() {
        var s = BsafeSettings()
        s.quality = .fast
        let fast = arguments(for: s)
        XCTAssertTrue(fast.contains("--extra-scales"))
        XCTAssertEqual(Array(fast.dropFirst(fast.firstIndex(of: "--extra-scales")! + 1).prefix(1)), ["none"])
        XCTAssertEqual(Array(fast.dropFirst(fast.firstIndex(of: "--detect-scale")! + 1).prefix(1)), ["1.0"])

        s.quality = .strict
        let strict = arguments(for: s)
        XCTAssertEqual(Array(strict.dropFirst(strict.firstIndex(of: "--extra-scales")! + 1).prefix(1)), ["0.5"])
        XCTAssertEqual(Array(strict.dropFirst(strict.firstIndex(of: "--detect-scale")! + 1).prefix(1)), ["1.5"])
    }

    func testFeetFlagAlwaysExplicit() {
        var s = BsafeSettings()
        s.feet = false
        XCTAssertTrue(arguments(for: s).contains("--no-feet"))
        XCTAssertFalse(arguments(for: s).contains("--feet"))
        s.feet = true
        XCTAssertTrue(arguments(for: s).contains("--feet"))
        XCTAssertFalse(arguments(for: s).contains("--no-feet"))
    }

    func testCensorValues() {
        for censor in [Censor.body, .all, .female, .male] {
            var s = BsafeSettings()
            s.censor = censor
            let args = arguments(for: s)
            let i = args.firstIndex(of: "--censor")!
            XCTAssertEqual(args[i + 1], censor.rawValue)
        }
    }

    func testPaddingFormatting() {
        XCTAssertEqual(formatPadding(0.4), "0.4")
        XCTAssertEqual(formatPadding(0), "0")
        XCTAssertEqual(formatPadding(0.6), "0.6")
        var s = BsafeSettings()
        s.padding = 0.1
        XCTAssertTrue(arguments(for: s).contains("0.1"))
    }

    // MARK: - stats parsing

    let sample = "live stats (window): frames=77 rate=38.1/s avg_detect_ms=22.4 max_detect_ms=31.2 avg_receive_to_send_ms=22.4"

    func testParseStatsLine() {
        let stats = parseStatsLine(sample)
        XCTAssertNotNil(stats)
        XCTAssertEqual(stats!.rate, 38.1, accuracy: 1e-9)
        XCTAssertEqual(stats!.detectMs, 22.4, accuracy: 1e-9)
    }

    func testParseStatsLineWithANSI() {
        let wrapped = "\u{1B}[32m" + sample + "\u{1B}[0m"
        let stats = parseStatsLine(wrapped)
        XCTAssertNotNil(stats)
        XCTAssertEqual(stats!.rate, 38.1, accuracy: 1e-9)
        XCTAssertEqual(stats!.detectMs, 22.4, accuracy: 1e-9)
    }

    func testParseStatsLineRejectsOthers() {
        XCTAssertNil(parseStatsLine("native: display=1 fps=42.5 queued=0 dropped=0"))
        XCTAssertNil(parseStatsLine("Running... press Ctrl+C to stop."))
        XCTAssertNil(parseStatsLine("Error: something broke"))
        XCTAssertNil(parseStatsLine("garbage line without numbers"))
        XCTAssertNil(parseStatsLine(""))
    }

    func testStripANSI() {
        XCTAssertEqual(stripANSI("\u{1B}[1;31mError:\u{1B}[0m boom"), "Error: boom")
        XCTAssertEqual(stripANSI("plain line"), "plain line")
    }

    // MARK: - settings persistence

    func testSettingsDefaults() {
        let s = BsafeSettings()
        XCTAssertEqual(s.censor, .body)
        XCTAssertTrue(s.feet)
        XCTAssertEqual(s.style, .pixelate)
        XCTAssertEqual(s.padding, 0.4, accuracy: 1e-9)
        XCTAssertEqual(s.minPadding, 24)
        XCTAssertEqual(s.quality, .standard)
    }

    func testSettingsCodableRoundTrip() throws {
        let s = BsafeSettings(censor: .female, feet: false, style: .blur, padding: 0.2, minPadding: 8, quality: .strict)
        let data = try JSONEncoder().encode(s)
        XCTAssertEqual(try JSONDecoder().decode(BsafeSettings.self, from: data), s)
    }

    // MARK: - executable resolution

    func testResolveExecutablePrefersOverride() {
        let url = resolveBsafeExecutable(infoPlistPath: "/nonexistent.plist", defaults: "/tmp/custom/bsafe")
        XCTAssertEqual(url?.path, "/tmp/custom/bsafe")
    }

    func testResolveExecutableFromPlistFile() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let plist = dir.appendingPathComponent("Info.plist")
        let content = """
            <?xml version="1.0" encoding="UTF-8"?>
            <plist version="1.0"><dict>
            <key>BsafeCLIPath</key><string>/tmp/repo/.venv/bin/bsafe</string>
            </dict></plist>
            """
        try content.write(to: plist, atomically: true, encoding: .utf8)
        XCTAssertEqual(
            resolveBsafeExecutable(infoPlistPath: plist.path, defaults: nil)?.path,
            "/tmp/repo/.venv/bin/bsafe"
        )
        XCTAssertNil(resolveBsafeExecutable(infoPlistPath: dir.appendingPathComponent("Missing.plist").path, defaults: nil))
        try? FileManager.default.removeItem(at: dir)
    }

    // MARK: - process supervision

    private let sh = URL(fileURLWithPath: "/bin/sh")
    private let tmp = URL(fileURLWithPath: "/tmp")

    func testProcessReachesRunningWithStatsAndStops() {
        let process = BsafeProcess()
        let running = expectation(description: "reaches running with stats")
        var fulfilled = false
        process.onStateChange = { state in
            if case .running(let maybeStats) = state, let stats = maybeStats, !fulfilled {
                fulfilled = true
                XCTAssertEqual(stats.rate, 10.0, accuracy: 1e-9)
                XCTAssertEqual(stats.detectMs, 5.0, accuracy: 1e-9)
                running.fulfill()
            }
        }
        XCTAssertTrue(process.start(
            executable: sh,
            arguments: ["-c", "echo Running...; echo \"live stats (window): frames=1 rate=10.0/s avg_detect_ms=5.0 max_detect_ms=5 avg_receive_to_send_ms=5\"; sleep 30"],
            cwd: tmp,
            env: [:]
        ))
        wait(for: [running], timeout: 5.0)

        let stopped = expectation(description: "stops after SIGINT")
        process.onStateChange = { state in
            if state == .stopped { stopped.fulfill() }
        }
        process.stop()
        wait(for: [stopped], timeout: 5.0)
        XCTAssertEqual(process.currentState, .stopped)
    }

    func testProcessEscalatesPastIgnoredSIGINT() {
        let process = BsafeProcess()
        let running = expectation(description: "reaches running")
        var fulfilled = false
        process.onStateChange = { state in
            if case .running = state, !fulfilled {
                fulfilled = true
                running.fulfill()
            }
        }
        XCTAssertTrue(process.start(
            executable: sh,
            arguments: ["-c", "trap \"\" INT; echo Running...; sleep 30"],
            cwd: tmp,
            env: [:]
        ))
        wait(for: [running], timeout: 5.0)

        let stopped = expectation(description: "escalates to SIGTERM and stops")
        process.onStateChange = { state in
            if state == .stopped { stopped.fulfill() }
        }
        let start = Date()
        process.stop()
        wait(for: [stopped], timeout: 12.0)
        let elapsed = Date().timeIntervalSince(start)
        // SIGINT is ignored, so stop() must wait out the 3 s escalation delay.
        XCTAssertGreaterThanOrEqual(elapsed, 2.5)
        XCTAssertLessThan(elapsed, 10.0)
        XCTAssertEqual(process.currentState, .stopped)
    }

    func testProcessReportsFailure() {
        let process = BsafeProcess()
        let failed = expectation(description: "reports failure")
        process.onStateChange = { state in
            if case .failed(let message) = state {
                XCTAssertEqual(message, "Error: boom")
                failed.fulfill()
            }
        }
        XCTAssertTrue(process.start(
            executable: sh,
            arguments: ["-c", "echo Error: boom; exit 1"],
            cwd: tmp,
            env: [:]
        ))
        wait(for: [failed], timeout: 5.0)
        XCTAssertEqual(process.currentState, .failed("Error: boom"))
    }

    func testTerminateNowKillsGroup() {
        let process = BsafeProcess()
        let running = expectation(description: "reaches running")
        var fulfilled = false
        process.onStateChange = { state in
            if case .running = state, !fulfilled {
                fulfilled = true
                running.fulfill()
            }
        }
        XCTAssertTrue(process.start(
            executable: sh,
            arguments: ["-c", "echo Running...; sleep 30 & wait"],
            cwd: tmp,
            env: [:]
        ))
        wait(for: [running], timeout: 5.0)
        let pgid = process.activeChildPid
        XCTAssertNotEqual(pgid, 0)
        XCTAssertEqual(kill(-pgid, 0), 0)
        process.terminateNow(timeout: 1.5)
        // The whole group (shell + sleeper child) must be gone.
        var groupGone = (kill(-pgid, 0) != 0)
        let deadline = Date().addingTimeInterval(2.0)
        while !groupGone, Date() < deadline {
            usleep(50_000)
            groupGone = (kill(-pgid, 0) != 0)
        }
        XCTAssertTrue(groupGone, "child process group survived terminateNow")
    }

    func testSpawnResetsParentIgnoredSIGINT() {
        // Ignore SIGINT in this test process; the child must still die on
        // SIGINT thanks to POSIX_SPAWN_SETSIGDEF (not inherit SIG_IGN).
        let previous = signal(SIGINT, SIG_IGN)
        defer {
            // Restore previous disposition (usually SIG_DFL in tests).
            if let previous {
                signal(SIGINT, previous)
            } else {
                signal(SIGINT, SIG_DFL)
            }
        }
        let process = BsafeProcess()
        let running = expectation(description: "reaches running")
        var fulfilled = false
        process.onStateChange = { state in
            if case .running = state, !fulfilled {
                fulfilled = true
                running.fulfill()
            }
        }
        XCTAssertTrue(process.start(
            executable: sh,
            arguments: ["-c", "echo Running...; sleep 30"],
            cwd: tmp,
            env: [:]
        ))
        wait(for: [running], timeout: 5.0)
        let stopped = expectation(description: "stops quickly on SIGINT")
        process.onStateChange = { state in
            if state == .stopped { stopped.fulfill() }
        }
        let start = Date()
        process.stop()
        wait(for: [stopped], timeout: 5.0)
        let elapsed = Date().timeIntervalSince(start)
        // Without SETSIGDEF the child would inherit SIG_IGN and need the
        // 3 s SIGTERM escalation; with it, SIGINT kills immediately.
        XCTAssertLessThan(elapsed, 2.5)
        XCTAssertEqual(process.currentState, .stopped)
    }
}
