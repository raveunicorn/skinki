// swift-tools-version: 6.0
import PackageDescription

// TextEngine — the Global Text Engine: rewrite / translate / summarize the
// user's selected text in any app, then replace it in place.
let package = Package(
    name: "TextEngine",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "TextEngine", targets: ["TextEngine"]),
    ],
    dependencies: [
        .package(path: "../SkinkiCore"),
        .package(path: "../SystemBridge"),
    ],
    targets: [
        .target(name: "TextEngine", dependencies: ["SkinkiCore", "SystemBridge"]),
        .testTarget(name: "TextEngineTests", dependencies: ["TextEngine"]),
    ],
    swiftLanguageModes: [.v5]
)
