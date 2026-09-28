import Foundation

public enum RunState: Equatable {
    case stopped
    case starting
    case running(LiveStats?)
    case stopping
    case failed(String)
}

/// Resolve the bsafe CLI executable. Priority: explicit user override
/// (UserDefaults key "bsafeExecutablePath") -> Info.plist key "BsafeCLIPath"
/// (read from the given plist path, or the main bundle when nil) -> nil.
public func resolveBsafeExecutable(infoPlistPath: String?, defaults `override`: String?) -> URL? {
    if let o = `override`?.trimmingCharacters(in: .whitespacesAndNewlines), !o.isEmpty {
        return URL(fileURLWithPath: (o as NSString).expandingTildeInPath)
    }
    var plistValue: String?
    if let path = infoPlistPath {
        if let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
           let plist = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any]
        {
            plistValue = plist["BsafeCLIPath"] as? String
        }
    } else {
        plistValue = Bundle.main.object(forInfoDictionaryKey: "BsafeCLIPath") as? String
    }
    if let v = plistValue?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty {
        return URL(fileURLWithPath: v)
    }
    return nil
}

/// Supervises one `bsafe start` child process.
///
/// The child is spawned with posix_spawn + POSIX_SPAWN_SETPGROUP so it becomes
/// a process-group leader (pgid == child pid); stop() signals the whole group
/// so the Swift overlay helper (a grandchild) dies too. stdout and stderr are
/// each drained on a dedicated background thread (a full pipe would block the
/// child and freeze the overlay); the last 50 lines are retained.
/// All onStateChange/onLine callbacks are invoked on the main queue.
public final class BsafeProcess {
    public var onStateChange: ((RunState) -> Void)?
    public var onLine: ((String) -> Void)?

    /// UserDefaults key holding the running child's pgid (== child pid,
    /// since the child is a process-group leader). Present only while a
    /// child is running; a stale value after a crash/quit means an orphan
    /// group may still be alive.
    public static let childPGIDDefaultsKey = "BsafeChildPGID"

    private let lock = NSLock()
    private var _state: RunState = .stopped
    private var childPid: pid_t = 0
    private var childActive = false
    private var userStopping = false
    private var storedLines: [String] = []
    private var escalation: [DispatchWorkItem] = []

    public init() {}

    public var currentState: RunState { lock.withLock { _state } }
    public var isRunning: Bool { lock.withLock { childActive } }
    public var recentLines: [String] { lock.withLock { storedLines } }

    /// Current child pid (== pgid while running), 0 when idle. For tests
    /// and synchronous termination.
    public var activeChildPid: pid_t { lock.withLock { childPid } }

    /// Must be called on the main queue.
    private func emit(_ state: RunState) {
        lock.withLock { _state = state }
        onStateChange?(state)
    }

    private func emitOnMain(_ state: RunState) {
        if Thread.isMainThread {
            emit(state)
        } else {
            DispatchQueue.main.async { self.emit(state) }
        }
    }

    @discardableResult
    public func start(executable: URL, arguments: [String], cwd: URL, env: [String: String]) -> Bool {
        if lock.withLock({ childActive }) { return false }

        var outPipe = [Int32](repeating: 0, count: 2)
        var errPipe = [Int32](repeating: 0, count: 2)
        guard pipe(&outPipe) == 0, pipe(&errPipe) == 0 else {
            emitOnMain(.failed("could not create pipes for bsafe"))
            return false
        }

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        posix_spawn_file_actions_adddup2(&actions, outPipe[1], STDOUT_FILENO)
        posix_spawn_file_actions_addclose(&actions, outPipe[1])
        posix_spawn_file_actions_adddup2(&actions, errPipe[1], STDERR_FILENO)
        posix_spawn_file_actions_addclose(&actions, errPipe[1])
        posix_spawn_file_actions_addclose(&actions, outPipe[0])
        posix_spawn_file_actions_addclose(&actions, errPipe[0])
        posix_spawn_file_actions_addchdir_np(&actions, cwd.path)

        var attrs: posix_spawnattr_t?
        posix_spawnattr_init(&attrs)
        posix_spawnattr_setflags(&attrs, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF))
        posix_spawnattr_setpgroup(&attrs, 0)
        // Reset ignored dispositions inherited from a GUI parent (notably
        // SIGINT) back to the default so stop()'s SIGINT step works.
        var sigdefault = sigset_t()
        sigemptyset(&sigdefault)
        sigaddset(&sigdefault, SIGINT)
        sigaddset(&sigdefault, SIGTERM)
        sigaddset(&sigdefault, SIGPIPE)
        posix_spawnattr_setsigdefault(&attrs, &sigdefault)

        let argv0 = executable.path
        let allArgs = [argv0] + arguments
        var argv: [UnsafeMutablePointer<CChar>?] = allArgs.map { strdup($0) }
        argv.append(nil)

        var mergedEnv = ProcessInfo.processInfo.environment
        for (k, v) in env { mergedEnv[k] = v }
        var envp: [UnsafeMutablePointer<CChar>?] = mergedEnv.map { strdup("\($0.key)=\($0.value)") }
        envp.append(nil)

        var pid: pid_t = 0
        let spawnResult: Int32 = argv.withUnsafeMutableBufferPointer { argvBuf in
            envp.withUnsafeMutableBufferPointer { envBuf in
                posix_spawn(&pid, argv0, &actions, &attrs, argvBuf.baseAddress!, envBuf.baseAddress!)
            }
        }

        for p in argv { if let p = p { free(p) } }
        for p in envp { if let p = p { free(p) } }
        posix_spawn_file_actions_destroy(&actions)
        posix_spawnattr_destroy(&attrs)

        guard spawnResult == 0 else {
            _ = close(outPipe[0])
            _ = close(outPipe[1])
            _ = close(errPipe[0])
            _ = close(errPipe[1])
            emitOnMain(.failed("could not launch bsafe: \(String(cString: strerror(spawnResult)))"))
            return false
        }

        // Parent keeps only the read ends; the child owns the write ends now,
        // so EOF arrives once the child (and any grandchildren) exit.
        _ = close(outPipe[1])
        _ = close(errPipe[1])

        lock.withLock {
            childPid = pid
            childActive = true
            userStopping = false
            storedLines = []
            escalation = []
        }
        UserDefaults.standard.set(Int(pid), forKey: Self.childPGIDDefaultsKey)
        emitOnMain(.starting)

        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global(qos: .utility).async { self.drainLoop(fd: outPipe[0], group: group) }
        group.enter()
        DispatchQueue.global(qos: .utility).async { self.drainLoop(fd: errPipe[0], group: group) }
        DispatchQueue.global(qos: .utility).async { self.waitLoop(pid: pid, group: group) }
        return true
    }

    /// SIGINT the group, then SIGTERM after 3 s, then SIGKILL after 3 more s.
    public func stop() {
        let pid: pid_t? = lock.withLock {
            if !childActive { return nil }
            userStopping = true
            return childPid
        }
        guard let pid = pid, pid != 0 else {
            if case .failed = currentState {} else {
                emitOnMain(.stopped)
            }
            return
        }
        emitOnMain(.stopping)
        _ = kill(-pid, SIGINT)
        let toTerm = DispatchWorkItem { [weak self] in
            guard let self = self, self.isRunning else { return }
            _ = kill(-pid, SIGTERM)
        }
        let toKill = DispatchWorkItem { [weak self] in
            guard let self = self, self.isRunning else { return }
            _ = kill(-pid, SIGKILL)
        }
        _ = lock.withLock { escalation = [toTerm, toKill] }
        DispatchQueue.global().asyncAfter(deadline: .now() + 3.0, execute: toTerm)
        DispatchQueue.global().asyncAfter(deadline: .now() + 6.0, execute: toKill)
    }

    /// Synchronously terminate the child group (for willTerminate).
    /// SIGTERM the group, poll waitpid(WNOHANG) for up to `timeout`
    /// seconds, then SIGKILL the group. Never blocks holding the lock.
    public func terminateNow(timeout: TimeInterval = 1.5) {
        let pid: pid_t? = lock.withLock { childActive ? childPid : nil }
        guard let pid = pid, pid != 0 else { return }
        lock.withLock {
            for item in escalation { item.cancel() }
            escalation = []
        }
        _ = kill(-pid, SIGTERM)
        let deadline = Date().addingTimeInterval(timeout)
        var reaped = false
        while Date() < deadline {
            var s: Int32 = 0
            let r = waitpid(pid, &s, WNOHANG)
            if r == pid {
                reaped = true
                break
            } else if r == 0 {
                usleep(50_000)
                continue
            } else {
                // Error (ECHILD means waitLoop already reaped).
                reaped = true
                break
            }
        }
        var check: Int32 = 0
        let still = waitpid(pid, &check, WNOHANG)
        if still == 0 && !reaped {
            _ = kill(-pid, SIGKILL)
        }
        // The leader exiting is not enough: other group members (the overlay
        // helper) must be gone too. Wait out the rest of the timeout, then
        // SIGKILL whatever is left and give it a short grace period.
        while kill(-pid, 0) == 0, Date() < deadline {
            usleep(50_000)
        }
        if kill(-pid, 0) == 0 {
            _ = kill(-pid, SIGKILL)
            let killDeadline = Date().addingTimeInterval(0.5)
            while kill(-pid, 0) == 0, Date() < killDeadline {
                usleep(20_000)
            }
        }
        lock.withLock {
            childActive = false
            userStopping = false
            for item in escalation { item.cancel() }
            escalation = []
        }
        UserDefaults.standard.removeObject(forKey: Self.childPGIDDefaultsKey)
    }

    deinit {
        let pid = lock.withLock { childPid }
        if lock.withLock({ childActive }), pid != 0 {
            _ = kill(-pid, SIGKILL)
        }
    }

    private func drainLoop(fd: Int32, group: DispatchGroup) {
        defer {
            group.leave()
            _ = close(fd)
        }
        var pending = Data()
        var chunk = [UInt8](repeating: 0, count: 4096)
        while true {
            let n: Int = chunk.withUnsafeMutableBytes { raw in
                guard let base = raw.baseAddress else { return -1 }
                return read(fd, base, 4096)
            }
            if n < 0 {
                if errno == EINTR { continue }
                break
            }
            if n == 0 { break }
            pending.append(contentsOf: chunk.prefix(n))
            while let nl = pending.firstIndex(of: 10) {
                let lineData = pending.subdata(in: pending.startIndex..<nl)
                pending.removeSubrange(pending.startIndex...nl)
                handleRawLine(lineData)
            }
        }
        if !pending.isEmpty {
            handleRawLine(pending)
        }
    }

    private func handleRawLine(_ data: Data) {
        var text = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) ?? ""
        if text.hasSuffix("\r") { text.removeLast() }
        let stats = parseStatsLine(text)
        let runningMarker = text.contains("Running...")
        lock.withLock {
            storedLines.append(text)
            if storedLines.count > 50 {
                storedLines.removeFirst(storedLines.count - 50)
            }
        }
        DispatchQueue.main.async {
            self.onLine?(text)
            switch self.currentState {
            case .starting:
                if let s = stats {
                    self.emit(.running(s))
                } else if runningMarker {
                    self.emit(.running(nil))
                }
            case .running:
                if let s = stats {
                    self.emit(.running(s))
                }
            default:
                break
            }
        }
    }

    private func waitLoop(pid: pid_t, group: DispatchGroup) {
        var status: Int32 = 0
        _ = waitpid(pid, &status, 0)
        // Let the readers consume remaining output (bounded wait).
        _ = group.wait(timeout: .now() + 2.0)
        DispatchQueue.main.async { self.finish(status: status) }
    }

    private func finish(status: Int32) {
        let wasStopping = lock.withLock { () -> Bool in
            let v = userStopping
            userStopping = false
            childActive = false
            for item in escalation { item.cancel() }
            escalation = []
            return v
        }
        UserDefaults.standard.removeObject(forKey: Self.childPGIDDefaultsKey)
        if wasStopping {
            emit(.stopped)
            return
        }
        let exitedCleanly = (status & 0x7F) == 0 && ((status >> 8) & 0xFF) == 0
        if exitedCleanly {
            emit(.stopped)
            return
        }
        let lines = lock.withLock { storedLines }
        let message = lines.last(where: { $0.hasPrefix("Error:") }) ?? lines.last ?? "bsafe exited unexpectedly"
        emit(.failed(message))
    }

    // MARK: - Orphan reclamation

    /// Kill a stale child group left by a previous app instance (crash,
    /// SIGKILL, or quit without reaping). Only kills when the stored pgid
    /// is still alive AND its command line belongs to bsafe; never kills
    /// an unrelated reused pid. Called on launch before any autoStart.
    public static func reclaimOrphanedGroupIfNeeded(defaults: UserDefaults = .standard) {
        guard let stored = defaults.object(forKey: childPGIDDefaultsKey) else { return }
        let pgidInt: Int? = (stored as? Int) ?? (stored as? NSNumber)?.intValue
        guard let raw = pgidInt, raw != 0 else {
            defaults.removeObject(forKey: childPGIDDefaultsKey)
            return
        }
        let pgid = pid_t(raw)
        // Group already gone: just clear the stale key.
        if kill(-pgid, 0) != 0 {
            defaults.removeObject(forKey: childPGIDDefaultsKey)
            return
        }
        guard groupContainsBsafe(pgid: pgid) else {
            defaults.removeObject(forKey: childPGIDDefaultsKey)
            return
        }
        _ = kill(-pgid, SIGTERM)
        let deadline = Date().addingTimeInterval(1.0)
        while Date() < deadline {
            if kill(-pgid, 0) != 0 { break }
            usleep(50_000)
        }
        if kill(-pgid, 0) == 0 {
            _ = kill(-pgid, SIGKILL)
        }
        defaults.removeObject(forKey: childPGIDDefaultsKey)
    }

    /// True when the leader's executable path contains "bsafe" (proc_pidpath)
    /// or any live member of the group has "bsafe" in its command line
    /// (covers `.venv/bin/bsafe` run through a python shebang, where the
    /// executable path is the interpreter).
    private static func groupContainsBsafe(pgid: pid_t) -> Bool {
        var buf = [CChar](repeating: 0, count: 4096)
        let resolved: String? = buf.withUnsafeMutableBufferPointer { ptr -> String? in
            guard let base = ptr.baseAddress else { return nil }
            let ret = proc_pidpath(pgid, base, UInt32(ptr.count))
            guard ret > 0 else { return nil }
            return String(cString: base)
        }
        if let resolved, resolved.contains("bsafe") { return true }
        if isBsafeCommandLine(pid: pgid) { return true }
        // Leader may be gone while a helper child still holds the group.
        // Scan for any member of the group with bsafe in its command line.
        return anyGroupMemberIsBsafe(pgid: pgid)
    }

    private static func isBsafeCommandLine(pid: pid_t) -> Bool {
        let ps = Process()
        ps.executableURL = URL(fileURLWithPath: "/bin/ps")
        ps.arguments = ["-o", "command=", "-p", "\(pid)"]
        let pipe = Pipe()
        ps.standardOutput = pipe
        ps.standardError = FileHandle.nullDevice
        do {
            try ps.run()
        } catch {
            return false
        }
        ps.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let out = String(data: data, encoding: .utf8) else { return false }
        return out.contains("bsafe")
    }

    private static func anyGroupMemberIsBsafe(pgid: pid_t) -> Bool {
        let ps = Process()
        ps.executableURL = URL(fileURLWithPath: "/bin/ps")
        ps.arguments = ["-ax", "-o", "pid,pgid,command"]
        let pipe = Pipe()
        ps.standardOutput = pipe
        ps.standardError = FileHandle.nullDevice
        do {
            try ps.run()
        } catch {
            return false
        }
        ps.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let out = String(data: data, encoding: .utf8) else { return false }
        for line in out.split(separator: "\n") {
            // Columns: PID PGID COMMAND (command may contain spaces).
            let parts = line.split(separator: " ", maxSplits: 2, omittingEmptySubsequences: true)
            guard parts.count == 3, let memberPGID = Int(parts[1]) else { continue }
            if memberPGID == Int(pgid), parts[2].contains("bsafe") { return true }
        }
        return false
    }
}
