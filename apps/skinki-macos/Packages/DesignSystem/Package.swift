// swift-tools-version: 6.0
import PackageDescription

// DesignSystem — design tokens, joy-design components, and the Rive mascot.
// The single source of truth for color, blur, motion, and the lizard.
//
// NOTE: verify the Rive package URL/version (rive-app/rive-ios, product
// "RiveRuntime") when implementation begins.
let package = Package(
    name: "DesignSystem",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "DesignSystem", targets: ["DesignSystem"]),
    ],
    dependencies: [
        .package(path: "../SkinkiCore"),
        .package(url: "https://github.com/rive-app/rive-ios", from: "6.0.0"),
    ],
    targets: [
        .target(
            name: "DesignSystem",
            dependencies: [
                "SkinkiCore",
                .product(name: "RiveRuntime", package: "rive-ios"),
            ]
        ),
        .testTarget(name: "DesignSystemTests", dependencies: ["DesignSystem"]),
    ],
    swiftLanguageModes: [.v5]
)
