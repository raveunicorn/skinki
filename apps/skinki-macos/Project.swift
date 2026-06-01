import ProjectDescription

// Skinki — the thin macOS app target.
//
// Functionality and third-party dependencies live in local Swift packages
// under `Packages/`. Tuist generates the Xcode project + workspace and wires
// the app target to those packages. See ARCHITECTURE.md for the module map.

let bundleId = "com.skinki.app"
let deploymentTarget = "15.0"

let baseSettings: SettingsDictionary = [
    "SWIFT_VERSION": "6.0",
    "MACOSX_DEPLOYMENT_TARGET": .string(deploymentTarget),
    // Non-sandboxed, Developer-ID, notarized distribution (see ARCHITECTURE.md §7).
    "ENABLE_HARDENED_RUNTIME": "YES",
    "CODE_SIGN_STYLE": "Automatic",
    "MARKETING_VERSION": "0.1.0",
    "CURRENT_PROJECT_VERSION": "1",
    "ENABLE_USER_SCRIPT_SANDBOXING": "NO",
]

let project = Project(
    name: "Skinki",
    organizationName: "Skinki",
    options: .options(
        defaultKnownRegions: ["en", "ru"],
        developmentRegion: "en"
    ),
    packages: [
        .local(path: "Packages/SkinkiCore"),
        .local(path: "Packages/InferenceEngine"),
        .local(path: "Packages/MemoryStore"),
        .local(path: "Packages/SystemBridge"),
        .local(path: "Packages/TextEngine"),
        .local(path: "Packages/VoiceEngine"),
        .local(path: "Packages/DesignSystem"),
        .local(path: "Packages/Features"),
    ],
    settings: .settings(base: baseSettings),
    targets: [
        .target(
            name: "Skinki",
            destinations: .macOS,
            product: .app,
            bundleId: bundleId,
            deploymentTargets: .macOS(deploymentTarget),
            infoPlist: .file(path: "App/Resources/Info.plist"),
            sources: ["App/Sources/**"],
            resources: ["App/Resources/Assets.xcassets/**"],
            entitlements: .file(path: "App/Resources/Skinki.entitlements"),
            dependencies: [
                .package(product: "Features"),
                .package(product: "DesignSystem"),
                .package(product: "SkinkiCore"),
            ],
            settings: .settings(base: baseSettings)
        ),
        .target(
            name: "SkinkiTests",
            destinations: .macOS,
            product: .unitTests,
            bundleId: "\(bundleId).tests",
            deploymentTargets: .macOS(deploymentTarget),
            infoPlist: .default,
            sources: ["App/Tests/**"],
            dependencies: [
                .target(name: "Skinki"),
            ]
        ),
    ]
)
