import AppKit
import Foundation

// ---------------------------------------------------------------------------
// Caption bar geometry
// ---------------------------------------------------------------------------
let captionBarHeight: CGFloat = 160   // taller to fit 3 lines
let captionBarMargin: CGFloat = 16

// ---------------------------------------------------------------------------
// AppDelegate
// ---------------------------------------------------------------------------
@MainActor
class AppDelegate: NSObject, NSApplicationDelegate {

    var overlayWindow: NSWindow?

    // Rolling 3-line caption history. Stored here so buildOverlay() can
    // re-apply the current text after a display-configuration change.
    var captionHistory: [String] = []
    let maxLines = 3

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildOverlay()
        startStdinReader()

        // Rebuild if display configuration changes (resolution, external monitor, etc.)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(screensDidChange),
            name: NSApplication.didChangeScreenParametersNotification,
            object: nil
        )
    }

    @objc func screensDidChange() {
        overlayWindow?.close()
        overlayWindow = nil
        buildOverlay()
    }

    // MARK: - Stdin reader

    /// Spawns a background task that reads caption lines from stdin (one line
    /// per translated segment from the Python pipeline) and updates the overlay.
    func startStdinReader() {
        Task.detached {
            while let line = readLine(strippingNewline: true) {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                await MainActor.run {
                    self.pushCaption(trimmed)
                }
            }
        }
    }

    /// Append a new caption line, keep the last `maxLines`, update the view.
    func pushCaption(_ line: String) {
        captionHistory.append(line)
        if captionHistory.count > maxLines {
            captionHistory.removeFirst(captionHistory.count - maxLines)
        }
        updateCaptionView()
    }

    func updateCaptionView() {
        let text = captionHistory.joined(separator: "\n")
        (overlayWindow?.contentView as? CaptionView)?.captionText = text
    }

    // MARK: - Build overlay

    func buildOverlay() {
        guard let screen = NSScreen.main else { return }

        let screenFrame = screen.frame
        let windowFrame = NSRect(
            x: screenFrame.minX,
            y: screenFrame.minY + captionBarMargin,
            width: screenFrame.width,
            height: captionBarHeight
        )

        let window = NSWindow(
            contentRect: windowFrame,
            styleMask: .borderless,
            backing: .buffered,
            defer: false
        )

        // --- Transparency -------------------------------------------------
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false

        // --- Always on top, including over full-screen apps ---------------
        window.level = .screenSaver
        window.collectionBehavior = [
            .canJoinAllSpaces,
            .fullScreenAuxiliary,
            .stationary,
            .ignoresCycle
        ]

        // --- Click-through ------------------------------------------------
        window.ignoresMouseEvents = true

        // --- Content view -------------------------------------------------
        let contentView = CaptionView(frame: NSRect(origin: .zero, size: windowFrame.size))

        // Re-apply current history (survives display-change rebuilds).
        // Shows placeholder only until the first real caption arrives.
        if captionHistory.isEmpty {
            contentView.captionText = "Waiting for captions…"
        } else {
            contentView.captionText = captionHistory.joined(separator: "\n")
        }

        window.contentView = contentView
        window.orderFrontRegardless()
        overlayWindow = window
    }
}

// ---------------------------------------------------------------------------
// CaptionView  — the entire visible bar
// ---------------------------------------------------------------------------
@MainActor
class CaptionView: NSView {

    var captionText: String = "" {
        didSet { label.stringValue = captionText }
    }

    private let backgroundView = NSVisualEffectView()
    private let label = NSTextField(wrappingLabelWithString: "")

    override init(frame: NSRect) {
        super.init(frame: frame)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        wantsLayer = true

        // --- Frosted/dark background pill --------------------------------
        backgroundView.material = .hudWindow
        backgroundView.blendingMode = .behindWindow
        backgroundView.state = .active
        backgroundView.wantsLayer = true
        backgroundView.layer?.cornerRadius = 16
        backgroundView.layer?.masksToBounds = true
        backgroundView.alphaValue = 0.85
        backgroundView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(backgroundView)

        // --- Caption label -----------------------------------------------
        label.isEditable = false
        label.isSelectable = false
        label.isBezeled = false
        label.drawsBackground = false
        label.textColor = .white
        label.font = .boldSystemFont(ofSize: 24)
        label.alignment = .center
        label.maximumNumberOfLines = 3      // 3-line rolling history
        label.lineBreakMode = .byWordWrapping

        let shadow = NSShadow()
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.8)
        shadow.shadowOffset = NSSize(width: 0, height: -1)
        shadow.shadowBlurRadius = 4
        label.shadow = shadow

        label.translatesAutoresizingMaskIntoConstraints = false
        addSubview(label)

        // --- Layout -------------------------------------------------------
        let hPad: CGFloat = 32
        let vPad: CGFloat = 12
        let sidePad: CGFloat = 40

        NSLayoutConstraint.activate([
            backgroundView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: sidePad),
            backgroundView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -sidePad),
            backgroundView.topAnchor.constraint(equalTo: topAnchor),
            backgroundView.bottomAnchor.constraint(equalTo: bottomAnchor),

            label.leadingAnchor.constraint(equalTo: backgroundView.leadingAnchor, constant: hPad),
            label.trailingAnchor.constraint(equalTo: backgroundView.trailingAnchor, constant: -hPad),
            label.topAnchor.constraint(equalTo: backgroundView.topAnchor, constant: vPad),
            label.bottomAnchor.constraint(equalTo: backgroundView.bottomAnchor, constant: -vPad),
        ])
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
MainActor.assumeIsolated {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let delegate = AppDelegate()
    app.delegate = delegate
    app.run()
}
