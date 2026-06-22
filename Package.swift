// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CaptionOverlay",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "CaptionOverlay",
            path: "Sources/CaptionOverlay"
        )
    ]
)
