import AppKit
import QuartzCore

/// Snapshot of censor style applied to one overlay frame.
struct CensorStyle {
    var blur: Double
    var pixels: Double
    var text: String?

    init(blur: Double = 0.0, pixels: Double = 0.0, text: String? = nil) {
        self.blur = blur
        self.pixels = pixels
        self.text = text
    }
}

/// One overlay frame in capture-pixel coordinates (top-left origin).
/// boxes are integer-rounded outward; images[i] is the pre-pixelated
/// CGImage for boxes[i] (nil = render fallback), used only when pixels > 0.
struct OverlayFrame {
    var boxes: [CGRect]
    var images: [CGImage?]
    var captureWidth: Int
    var captureHeight: Int
    var blur: Double = 0.0
    var pixels: Double = 0.0
    var text: String? = nil
}

/// Run `body` on the main thread, calling directly when already there.
/// Startup runs on the main thread (an unconditional `main.sync` would
/// deadlock); restarts run on a background queue and must hop to main
/// before touching NSWindow/NSScreen.
func mainThreadSync<T>(_ body: @escaping () -> T) -> T {
    if Thread.isMainThread {
        return body()
    } else {
        return DispatchQueue.main.sync(execute: body)
    }
}

/// Snapshot overlay window-server ids without ever touching NSWindow off
/// the main thread.
func snapshotOverlayWindowIDs(_ overlays: [CensorOverlay]) -> Set<CGWindowID> {
    return mainThreadSync {
        Set(overlays.compactMap { $0.windowID })
    }
}

/// Borderless, transparent, always-on-top overlay window for censoring.
/// publish() is callable from any thread; rendering happens on the main
/// thread via a coalesced drain (latest frame wins, no backlog).
/// CALayer / NSVisualEffectView / CATextLayer pools are reused across
/// frames and never recreated per frame. No screen capture is performed here.
final class CensorOverlay {
    private var window: NSWindow?
    /// Window-server id of the overlay window (for excluding it from our own capture).
    /// Must only be touched on the main thread — restarts must use
    /// `safeWindowID()` / `snapshotOverlayWindowIDs(_:)` instead.
    var windowID: CGWindowID? { window.map { CGWindowID($0.windowNumber) } }

    /// Thread-safe window id snapshot (hops to main when off-main).
    /// Never touch `windowID` directly off the main thread.
    func safeWindowID() -> CGWindowID? {
        return mainThreadSync { self.windowID }
    }

    /// Re-read the NSScreen frame for `displayID` and move the overlay when
    /// the origin/size in points changed (e.g. arrangement change). Safe to
    /// call from any thread (hops to main internally). Pixel-size changes are
    /// handled by the caller's capture-size check (exit 3).
    func refreshFrame(displayID: CGDirectDisplayID) {
        mainThreadSync {
            let screenFrame: NSRect
            if let match = NSScreen.screens.first(where: {
                ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID) == displayID
            }) {
                screenFrame = match.frame
            } else {
                let cgBounds = CGDisplayBounds(displayID)
                let primaryHeight = NSScreen.screens.first?.frame.height ?? cgBounds.height
                screenFrame = NSRect(
                    x: cgBounds.origin.x,
                    y: primaryHeight - cgBounds.origin.y - cgBounds.height,
                    width: cgBounds.width,
                    height: cgBounds.height
                )
            }
            if screenFrame.origin != self.displayBounds.origin || screenFrame.size != self.displayBounds.size {
                self.displayBounds = screenFrame
                self.window?.setFrame(screenFrame, display: true)
            }
        }
    }
    private var contentView: NSView?
    private var displayBounds: CGRect = .zero

    private var boxLayers: [CALayer] = []
    private var blurViews: [NSVisualEffectView] = []
    private var textLayers: [CATextLayer] = []

    private let lock = NSLock()
    private var pending: OverlayFrame?
    private var drainScheduled = false
    private var drainCount = 0

    /// Create the overlay window covering the given display. Must be called on the main thread.
    func setup(displayID: CGDirectDisplayID, capturable: Bool) {
        // Use NSScreen.frame (AppKit coordinates, bottom-left origin) for window positioning.
        let screenFrame: NSRect
        if let match = NSScreen.screens.first(where: {
            ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID) == displayID
        }) {
            screenFrame = match.frame
        } else {
            // Fallback: convert CGDisplayBounds CG coords -> AppKit coords
            let cgBounds = CGDisplayBounds(displayID)
            let primaryHeight = NSScreen.screens.first?.frame.height ?? cgBounds.height
            screenFrame = NSRect(
                x: cgBounds.origin.x,
                y: primaryHeight - cgBounds.origin.y - cgBounds.height,
                width: cgBounds.width,
                height: cgBounds.height
            )
        }
        displayBounds = screenFrame

        let window = NSWindow(
            contentRect: screenFrame,
            styleMask: .borderless,
            backing: .buffered,
            defer: false
        )
        window.level = .screenSaver
        window.isOpaque = false
        window.backgroundColor = .clear
        window.ignoresMouseEvents = true
        // Normally excluded from ScreenCaptureKit (prevents feedback loop).
        // When capturable, visible to screenshots/recorders; our own capture
        // filter excludes our app instead (see DisplayPipeline startup).
        window.sharingType = capturable ? .readOnly : .none
        window.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]
        window.hasShadow = false

        let view = NSView(frame: NSRect(origin: .zero, size: screenFrame.size))
        view.wantsLayer = true
        window.contentView = view

        window.orderFrontRegardless()

        self.window = window
        self.contentView = view
    }

    /// Publish a frame from any thread. Coalesces to a single main-thread drain.
    func publish(_ frame: OverlayFrame) {
        lock.lock()
        pending = frame
        let shouldSchedule = !drainScheduled
        if shouldSchedule {
            drainScheduled = true
        }
        lock.unlock()
        if shouldSchedule {
            DispatchQueue.main.async { [weak self] in self?.drain() }
        }
    }

    /// Number of main-thread drains since the last call; resets the counter.
    func takeDrainCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        let n = drainCount
        drainCount = 0
        return n
    }

    private func drain() {
        lock.lock()
        let frame = pending
        pending = nil
        drainScheduled = false
        drainCount += 1
        lock.unlock()

        guard let frame else { return }
        guard let view = contentView, let rootLayer = view.layer else { return }
        guard frame.captureWidth > 0, frame.captureHeight > 0 else { return }

        let displayW = displayBounds.width
        let displayH = displayBounds.height
        guard displayW > 0, displayH > 0 else { return }

        // Convert capture-px boxes (top-left origin) to display points (bottom-left origin).
        let scaleX = displayW / CGFloat(frame.captureWidth)
        let scaleY = displayH / CGFloat(frame.captureHeight)
        let maxW = displayW * 0.8
        let maxH = displayH * 0.8
        var rects: [CGRect] = []
        rects.reserveCapacity(frame.boxes.count)
        for b in frame.boxes {
            var w = b.width * scaleX
            var h = b.height * scaleY
            if w > maxW { w = maxW }
            if h > maxH { h = maxH }
            let x = b.origin.x * scaleX
            let flippedY = displayH - b.origin.y * scaleY - h
            rects.append(CGRect(x: x, y: flippedY, width: w, height: h))
        }
        let n = rects.count

        let usePixels = frame.pixels > 0.0
        let useBlur = !usePixels && frame.blur > 0.0
        let backingScale = window?.backingScaleFactor ?? 2.0
        let blackCG = NSColor.black.cgColor
        let whiteCG = NSColor.white.cgColor

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        // Solid / pixelated box layers (hidden in blur mode).
        while boxLayers.count < n {
            let layer = CALayer()
            layer.isHidden = true
            layer.contentsGravity = .resize
            rootLayer.addSublayer(layer)
            boxLayers.append(layer)
        }
        for i in 0..<boxLayers.count {
            let layer = boxLayers[i]
            if i < n, !useBlur {
                layer.isHidden = false
                layer.frame = rects[i]
                if usePixels {
                    if i < frame.images.count, let img = frame.images[i] {
                        layer.contents = img
                    } else {
                        layer.contents = nil
                    }
                } else {
                    layer.contents = nil
                }
                layer.backgroundColor = blackCG
            } else {
                layer.isHidden = true
            }
        }

        // Blur views, reused and repositioned.
        if useBlur {
            while blurViews.count < n {
                let effect = NSVisualEffectView()
                effect.blendingMode = .behindWindow
                effect.material = .fullScreenUI
                effect.state = .active
                effect.isHidden = true
                view.addSubview(effect)
                blurViews.append(effect)
            }
            for i in 0..<blurViews.count {
                let effect = blurViews[i]
                if i < n {
                    effect.isHidden = false
                    effect.frame = rects[i]
                    effect.alphaValue = CGFloat(frame.blur)
                } else {
                    effect.isHidden = true
                }
            }
        } else {
            for effect in blurViews {
                effect.isHidden = true
            }
        }

        // Text labels, one pooled CATextLayer per box.
        if let text = frame.text, n > 0 {
            while textLayers.count < n {
                let layer = CATextLayer()
                layer.isHidden = true
                layer.alignmentMode = .center
                layer.isWrapped = false
                rootLayer.addSublayer(layer)
                textLayers.append(layer)
            }
            let attrsFont = NSFont.boldSystemFont(ofSize: 12)
            for i in 0..<textLayers.count {
                let layer = textLayers[i]
                if i < n {
                    layer.isHidden = false
                    let r = rects[i]
                    let fontSize = min(r.height * 0.3, 48)
                    let font = fontSize == 12 ? attrsFont : NSFont.boldSystemFont(ofSize: fontSize)
                    let attrs: [NSAttributedString.Key: Any] = [.font: font]
                    let textSize = (text as NSString).size(withAttributes: attrs)
                    let th = min(textSize.height, r.height)
                    layer.frame = CGRect(
                        x: r.origin.x,
                        y: r.origin.y + (r.height - th) / 2.0,
                        width: r.width,
                        height: th
                    )
                    layer.string = text
                    layer.font = font
                    layer.fontSize = fontSize
                    layer.foregroundColor = whiteCG
                    layer.contentsScale = backingScale
                } else {
                    layer.isHidden = true
                }
            }
        } else {
            for layer in textLayers {
                layer.isHidden = true
            }
        }

        CATransaction.commit()
    }
}
