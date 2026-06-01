import Foundation
import SkinkiCore
import SystemBridge

/// Orchestrates the Global Text Engine flow:
/// read selection -> run LLM transform -> replace in place.
public struct TextEngine: Sendable {
    private let llm: LLMService
    private let selection: SelectionReading
    private let input: InputSimulating

    public init(llm: LLMService, selection: SelectionReading, input: InputSimulating) {
        self.llm = llm
        self.selection = selection
        self.input = input
    }

    /// Stream the transformed text for the current selection (does not replace).
    public func transformSelection(
        action: TextAction,
        targetLanguage: String? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                guard let selected = await selection.readSelectedText(), !selected.isEmpty else {
                    continuation.finish(throwing: TextEngineError.noSelection)
                    return
                }
                let request = GenerationRequest(
                    system: Prompts.system(for: action, targetLanguage: targetLanguage),
                    messages: [ChatMessage(role: .user, content: selected)]
                )
                do {
                    for try await delta in llm.generate(request) {
                        continuation.yield(delta)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Transform and replace the selection in place.
    public func applyToSelection(action: TextAction, targetLanguage: String? = nil) async throws {
        var result = ""
        for try await delta in transformSelection(action: action, targetLanguage: targetLanguage) {
            result += delta
        }
        await input.replaceSelection(with: result)
    }
}

public enum TextEngineError: Error, Sendable {
    case noSelection
}
