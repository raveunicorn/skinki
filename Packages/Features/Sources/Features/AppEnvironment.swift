import Foundation
import Observation
import SkinkiCore
import InferenceEngine
import MemoryStore
import SystemBridge
import VoiceEngine
import DesignSystem

/// The composition root / dependency container. Built once at launch and
/// injected into the SwiftUI environment. Wires concrete implementations to the
/// `SkinkiCore` protocols so features depend only on abstractions.
@MainActor
@Observable
public final class AppEnvironment {
    public let inference: InferenceEngine
    public let memory: MemoryStore
    public let synthesizer: SpeechSynthesizing
    public let dictation: DictationService
    public let hotkeys: HotkeyCenter
    public let accessibility: AccessibilityBridge
    public let mascot: MascotController

    public let hardwareTier: HardwareTier
    public var selectedModel: ModelTier

    public init() {
        let engine = InferenceEngine()
        self.inference = engine
        self.memory = MemoryStore(embedder: engine)
        self.synthesizer = SystemSpeechSynthesizer()
        self.dictation = Dictation()
        self.hotkeys = HotkeyCenter()
        self.accessibility = AccessibilityBridge()
        self.mascot = MascotController()

        let tier = HardwareDetector.detect()
        self.hardwareTier = tier
        self.selectedModel = ModelTier.recommended(for: tier)
    }
}
