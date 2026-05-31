// swift-tools-version: 6.0
import PackageDescription

// SkinkiCore — domain models, cross-cutting protocols, DI, config, logging.
// The dependency sink: every other package depends on this, it depends on none.
let package = Package(
    name: "SkinkiCore",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "SkinkiCore", targets: ["SkinkiCore"]),
    ],
    targets: [
        .target(name: "SkinkiCore"),
        .testTarget(name: "SkinkiCoreTests", dependencies: ["SkinkiCore"]),
    ],
    swiftLanguageModes: [.v5]
)
