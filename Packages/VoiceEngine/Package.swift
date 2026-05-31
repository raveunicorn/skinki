// swift-tools-version: 6.0
import PackageDescription

// VoiceEngine — speech-to-text (native dictation) and text-to-speech.
// MVP TTS uses AVSpeechSynthesizer (good RU/EN Premium voices); the
// SpeechSynthesizing seam lets us swap in a local neural TTS later.
let package = Package(
    name: "VoiceEngine",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VoiceEngine", targets: ["VoiceEngine"]),
    ],
    dependencies: [
        .package(path: "../SkinkiCore"),
    ],
    targets: [
        .target(name: "VoiceEngine", dependencies: ["SkinkiCore"]),
        .testTarget(name: "VoiceEngineTests", dependencies: ["VoiceEngine"]),
    ],
    swiftLanguageModes: [.v5]
)
