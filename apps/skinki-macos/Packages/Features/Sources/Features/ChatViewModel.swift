import Foundation
import Observation
import SkinkiCore
import DesignSystem

/// Drives the chat conversation and the mascot's reactions.
@MainActor
@Observable
public final class ChatViewModel {
    public private(set) var messages: [ChatMessage] = []
    public private(set) var isGenerating = false
    public var input: String = ""

    private let llm: LLMService
    private let model: ModelTier
    private let mascot: MascotController
    private var task: Task<Void, Never>?

    public init(llm: LLMService, model: ModelTier, mascot: MascotController) {
        self.llm = llm
        self.model = model
        self.mascot = mascot
    }

    public func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating else { return }
        input = ""
        messages.append(ChatMessage(role: .user, content: text))

        var reply = ChatMessage(role: .assistant, content: "")
        messages.append(reply)
        isGenerating = true
        mascot.think()

        task = Task {
            do {
                try await llm.ensureLoaded(tier: model)
                let request = GenerationRequest(messages: messages.dropLast().map { $0 })
                mascot.talk()
                for try await delta in llm.generate(request) {
                    reply.content += delta
                    if let idx = messages.lastIndex(where: { $0.id == reply.id }) {
                        messages[idx] = reply
                    }
                }
                mascot.celebrate()
            } catch {
                Log.ui.error("Generation failed: \(error.localizedDescription, privacy: .public)")
                mascot.fail()
            }
            isGenerating = false
            mascot.rest()
        }
    }

    public func cancel() {
        task?.cancel()
        isGenerating = false
        mascot.rest()
    }
}
