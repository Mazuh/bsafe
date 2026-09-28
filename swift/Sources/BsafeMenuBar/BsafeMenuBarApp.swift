import AppKit
import SwiftUI

/// Holds the Controller for willTerminate without a Sendable closure.
/// NotificationCenter's selector API keeps only a weak reference to the
/// observer, so the App retains this handler for its lifetime.
private final class WillTerminateHandler: NSObject {
    let controller: BsafeController
    init(_ controller: BsafeController) {
        self.controller = controller
        super.init()
    }

    @objc func handleTermination(_ note: Notification) {
        MainActor.assumeIsolated {
            controller.terminateForWillTerminate()
        }
    }
}

@main
struct BsafeMenuBarApp: App {
    @StateObject private var controller: BsafeController

    private static var willTerminateHandler: WillTerminateHandler?

    @MainActor init() {
        _ = NSApplication.shared.setActivationPolicy(.accessory)
        let c = BsafeController()
        _controller = StateObject(wrappedValue: c)
        // Synchronously reap the CLI/helper group on any termination
        // (Cmd-Q, osascript quit, SIGTERM). Quit button keeps its graceful
        // path; this is the backstop that prevents orphans.
        let handler = WillTerminateHandler(c)
        Self.willTerminateHandler = handler
        NotificationCenter.default.addObserver(
            handler,
            selector: #selector(WillTerminateHandler.handleTermination(_:)),
            name: NSApplication.willTerminateNotification,
            object: nil
        )
    }

    var body: some Scene {
        MenuBarExtra("Bsafe", systemImage: controller.menuIconName) {
            PanelView(controller: controller)
        }
        .menuBarExtraStyle(.window)
    }
}
