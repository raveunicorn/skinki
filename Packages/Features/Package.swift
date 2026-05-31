// swift-tools-version: 6.0
import PackageDescription

// Features — composition layer. Wires the core packages into the SwiftUI
// surfaces: ChatHUD, MenuBarUI, Onboarding, Settings.
let package = Package(
    name: "Features",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "Features", targets: ["Features"]),
    ],
    dependencies: [
        .package(path: "../SkinkiCore"),
        .package(path: "../InferenceEngine"),
        .package(path: "../MemoryStore"),
        .package(path: "../SystemBridge"),
        .package(path: "../TextEngine"),
        .package(path: "../VoiceEngine"),
        .package(path: "../DesignSystem"),
    ],
    targets: [
        .target(
            name: "Features",
            dependencies: [
                "SkinkiCore",
                "InferenceEngine",
                "MemoryStore",
                "SystemBridge",
                "TextEngine",
                "VoiceEngine",
                "DesignSystem",
            ]
        ),
        .testTarget(name: "FeaturesTests", dependencies: ["Features"]),
    ],
    swiftLanguageModes: [.v5]
)
