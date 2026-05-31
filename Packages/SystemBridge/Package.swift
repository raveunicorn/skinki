// swift-tools-version: 6.0
import PackageDescription

// SystemBridge — the ONLY place allowed to touch fragile / permission-sensitive
// macOS APIs: Accessibility (selection + input simulation), global hotkeys,
// clipboard, shell, and (later) ScreenCaptureKit. See ARCHITECTURE.md §4.4.
let package = Package(
    name: "SystemBridge",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "SystemBridge", targets: ["SystemBridge"]),
    ],
    dependencies: [
        .package(path: "../SkinkiCore"),
    ],
    targets: [
        .target(name: "SystemBridge", dependencies: ["SkinkiCore"]),
        .testTarget(name: "SystemBridgeTests", dependencies: ["SystemBridge"]),
    ],
    swiftLanguageModes: [.v5]
)
