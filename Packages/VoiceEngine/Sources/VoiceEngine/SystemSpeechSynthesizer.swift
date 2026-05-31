import Foundation
import AVFoundation
import SkinkiCore

/// MVP text-to-speech using `AVSpeechSynthesizer`. Prefers Premium/Enhanced
/// voices for the target language (good Russian and English quality).
public final class SystemSpeechSynthesizer: SpeechSynthesizing, @unchecked Sendable {
    private let synthesizer = AVSpeechSynthesizer()

    public init() {}

    public func speak(_ text: String, language: String?) async {
        let utterance = AVSpeechUtterance(string: text)
        if let voice = Self.bestVoice(for: language) {
            utterance.voice = voice
        }
        synthesizer.speak(utterance)
    }

    public func stop() async {
        synthesizer.stopSpeaking(at: .immediate)
    }

    /// Pick the highest-quality installed voice for a language code (e.g. "ru-RU").
    static func bestVoice(for language: String?) -> AVSpeechSynthesisVoice? {
        let lang = language ?? AVSpeechSynthesisVoice.currentLanguageCode()
        let candidates = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix(String(lang.prefix(2))) }
        // Prefer Premium > Enhanced > Default.
        return candidates.max { lhs, rhs in lhs.quality.rawValue < rhs.quality.rawValue }
    }
}
