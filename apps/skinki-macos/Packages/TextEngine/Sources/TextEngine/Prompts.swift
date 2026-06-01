import Foundation

/// The text transformations the Global Text Engine offers.
public enum TextAction: String, Sendable, CaseIterable, Identifiable {
    case rewrite
    case improve
    case shorten
    case translate
    case summarize
    case fixGrammar

    public var id: String { rawValue }
}

/// Prompt templates. Skinki replies in the same language as the input unless a
/// target language is specified (full RU/EN parity is a requirement).
public enum Prompts {
    public static func system(for action: TextAction, targetLanguage: String? = nil) -> String {
        switch action {
        case .rewrite:
            return "Rewrite the user's text to be clearer and more natural. Preserve meaning and language. Output only the rewritten text."
        case .improve:
            return "Improve the user's text (clarity, flow, word choice). Keep the same language. Output only the improved text."
        case .shorten:
            return "Make the user's text more concise without losing key information. Same language. Output only the result."
        case .translate:
            let lang = targetLanguage ?? "English"
            return "Translate the user's text into \(lang). Output only the translation."
        case .summarize:
            return "Summarize the user's text in a few sentences, in the same language. Output only the summary."
        case .fixGrammar:
            return "Fix spelling, grammar, and punctuation in the user's text. Keep the same language and tone. Output only the corrected text."
        }
    }
}
