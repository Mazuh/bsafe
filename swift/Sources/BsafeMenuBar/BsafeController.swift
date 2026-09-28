import AppKit
import BsafeMenuKit
import ServiceManagement
import SwiftUI

@MainActor
final class BsafeController: ObservableObject {
    @Published private(set) var settings: BsafeSettings
    @Published private(set) var runState: RunState = .stopped
    @Published private(set) var loginEnabled = false
    @Published private(set) var loginError: String?

    private let process = BsafeProcess()
    private var restartTask: Task<Void, Never>?

    /// Set while Quit is in progress so state callbacks stop rewriting autoStart.
    private var quitting = false

    /// True while the user wants censoring on; an unexpected exit is then recovered.
    private var wantsRunning = false
    private var recoveryTask: Task<Void, Never>?
    private var recentFailures: [Date] = []
    private static let recoveryDelays: [UInt64] = [2, 5, 10, 30]

    init() {
        if let data = UserDefaults.standard.data(forKey: "BsafeSettings"),
           let decoded = try? JSONDecoder().decode(BsafeSettings.self, from: data)
        {
            settings = decoded
        } else {
            settings = BsafeSettings()
        }
        // Reap any orphaned CLI/helper group from a previous instance
        // before auto-starting a new one.
        BsafeProcess.reclaimOrphanedGroupIfNeeded()
        process.onStateChange = { [weak self] state in
            // BsafeProcess already hops to the main queue; re-enter the actor.
            Task { @MainActor [weak self] in self?.applyState(state) }
        }
        refreshLoginStatus()
        if UserDefaults.standard.bool(forKey: "autoStart") {
            Task { @MainActor [weak self] in self?.start() }
        }
    }

    var isActive: Bool {
        switch runState {
        case .running, .starting, .stopping: true
        case .stopped, .failed: false
        }
    }

    var menuIconName: String {
        switch runState {
        case .failed: "exclamationmark.triangle"
        case .running, .starting, .stopping: "eye.slash"
        case .stopped: "eye"
        }
    }

    var statusText: String {
        switch runState {
        case .stopped: return "Stopped"
        case .starting: return "Starting…"
        case .stopping: return "Stopping…"
        case .failed(let message): return message
        case .running(let stats):
            guard let stats else { return "Running…" }
            return "Running · \(Int(stats.rate.rounded()))/s · \(Int(stats.detectMs.rounded())) ms"
        }
    }

    var statusTint: Color {
        switch runState {
        case .running: .green
        case .failed: .red
        default: .secondary
        }
    }

    func binding<T>(_ keyPath: WritableKeyPath<BsafeSettings, T>) -> Binding<T> {
        Binding(get: { self.settings[keyPath: keyPath] }, set: { self.update(keyPath, $0) })
    }

    func update<T>(_ keyPath: WritableKeyPath<BsafeSettings, T>, _ value: T) {
        settings[keyPath: keyPath] = value
        if let data = try? JSONEncoder().encode(settings) {
            UserDefaults.standard.set(data, forKey: "BsafeSettings")
        }
        scheduleRestart()
    }

    var feetBinding: Binding<Bool> {
        Binding(
            // Body already includes feet, so show the toggle checked.
            get: { self.settings.censor == .body || self.settings.feet },
            set: { self.update(\.feet, $0) }
        )
    }

    func toggleRunning() {
        if process.isRunning {
            stop()
        } else {
            start()
        }
    }

    func start() {
        wantsRunning = true
        launch()
    }

    private func launch() {
        guard !process.isRunning else { return }
        guard let exe = resolveBsafeExecutable(
            infoPlistPath: nil,
            defaults: UserDefaults.standard.string(forKey: "bsafeExecutablePath")
        ) else {
            runState = .failed("bsafe CLI not found; reinstall with scripts/install-menubar.sh")
            return
        }
        // The venv executable is <repo>/.venv/bin/bsafe, so the repo root is
        // three levels up. Fall back to the executable's own directory.
        var cwd = exe.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var isDir: ObjCBool = false
        if !(FileManager.default.fileExists(atPath: cwd.path, isDirectory: &isDir) && isDir.boolValue) {
            cwd = exe.deletingLastPathComponent()
        }
        runState = .starting
        process.start(
            executable: exe,
            arguments: arguments(for: settings),
            cwd: cwd,
            env: ["NO_COLOR": "1"]
        )
    }

    func stop() {
        wantsRunning = false
        recoveryTask?.cancel()
        recoveryTask = nil
        restartTask?.cancel()
        restartTask = nil
        runState = .stopping
        process.stop()
    }

    func quit() {
        quitting = true
        UserDefaults.standard.set(process.isRunning || wantsRunning, forKey: "autoStart")
        wantsRunning = false
        recoveryTask?.cancel()
        restartTask?.cancel()
        restartTask = nil
        guard process.isRunning else {
            NSApplication.shared.terminate(nil)
            return
        }
        process.stop()
        Task { @MainActor [weak self] in
            guard let self else { return }
            let deadline = Date().addingTimeInterval(6.5)
            while self.process.isRunning, Date() < deadline {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            NSApplication.shared.terminate(nil)
        }
    }

    /// Synchronous group kill for willTerminate (Quit button keeps its
    /// graceful async path in quit()).
    func terminateForWillTerminate() {
        process.terminateNow(timeout: 1.5)
    }

    func refreshLoginStatus() {
        loginEnabled = SMAppService.mainApp.status == .enabled
    }

    func setLoginEnabled(_ enabled: Bool) {
        loginError = nil
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
        } catch {
            loginError = error.localizedDescription
        }
        refreshLoginStatus()
    }

    var qualityCaption: String {
        switch settings.quality {
        case .fast: "Lower CPU, may miss small regions."
        case .standard: "Balanced for everyday use."
        case .strict: "Higher CPU, catches smaller regions."
        }
    }

    private func applyState(_ state: RunState) {
        runState = state
        guard !quitting else { return }
        switch state {
        case .running:
            UserDefaults.standard.set(true, forKey: "autoStart")
        case .failed:
            if wantsRunning {
                scheduleRecovery()
            } else {
                UserDefaults.standard.set(false, forKey: "autoStart")
            }
        case .stopped:
            if wantsRunning {
                scheduleRecovery()
            } else {
                UserDefaults.standard.set(false, forKey: "autoStart")
            }
        case .starting, .stopping:
            break
        }
    }

    /// Relaunch after an unexpected exit: 2/5/10/30 s backoff, give up after
    /// 5 failures within 5 minutes (the failure message stays visible).
    private func scheduleRecovery() {
        let now = Date()
        recentFailures = recentFailures.filter { now.timeIntervalSince($0) < 300 } + [now]
        guard recentFailures.count < 5 else {
            wantsRunning = false
            UserDefaults.standard.set(false, forKey: "autoStart")
            return
        }
        let delay = Self.recoveryDelays[min(recentFailures.count - 1, Self.recoveryDelays.count - 1)]
        recoveryTask?.cancel()
        recoveryTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: delay * 1_000_000_000)
            guard let self, !Task.isCancelled, self.wantsRunning, !self.process.isRunning else { return }
            self.launch()
        }
    }

    /// Restart after a settings change, debounced 0.6 s.
    private func scheduleRestart() {
        guard process.isRunning else { return }
        restartTask?.cancel()
        restartTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 600_000_000)
            guard let self, !Task.isCancelled else { return }
            self.restartNow()
        }
    }

    private func restartNow() {
        process.stop()
        Task { @MainActor [weak self] in
            guard let self else { return }
            let deadline = Date().addingTimeInterval(7.0)
            while self.process.isRunning, Date() < deadline {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            if !Task.isCancelled, self.wantsRunning {
                self.launch()
            }
        }
    }
}
