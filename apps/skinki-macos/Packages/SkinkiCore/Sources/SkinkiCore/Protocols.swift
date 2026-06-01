import Foundation

// MARK: - Cross-cutting service protocols
//
// These abstractions let the UI and feature code depend on capabilities, not on
// concrete backends (MLX, SQLite, AVFoundation, macOS APIs). Concrete types live
// in their respective packages and are injected via `AppEnvironment`.

/// On-device large language model service. Implemented by `InferenceEngine`.
public protocol LLMService: Sendable {
    /// Ensure the given model tier is loaded (lazily, via mmap).
    func ensureLoaded(tier: ModelTier) async throws
    /// Release model weights back to the OS (idle / memory pressure).
    func unload() async
    /// Stream the assistant's reply as token deltas.
    func generate(_ request: GenerationRequest) -> AsyncThrowingStream<String, Error>
}

/// Text embedding service (EmbeddingGemma). Implemented by `InferenceEngine`.
public protocol EmbeddingService: Sendable {
    func embed(_ text: String) async throws -> [Float]
}

/// Long-term memory & RAG store. Implemented by `MemoryStore`.
public protocol MemoryStoring: Sendable {
    func remember(_ memory: NewMemory) async throws
    func retrieve(for query: String, limit: Int) async throws -> [RetrievedMemory]
    func pinned() async throws -> [Memory]
    func forget(id: Memory.ID) async throws
    func forgetAll() async throws
}

/// Text-to-speech. Implemented by `VoiceEngine` (AVSpeech now, neural later).
public protocol SpeechSynthesizing: Sendable {
    func speak(_ text: String, language: String?) async
    func stop() async
}

/// Speech-to-text dictation. Implemented by `VoiceEngine`.
public protocol DictationService: Sendable {
    /// Streams partial transcripts while the user speaks.
    func transcribe() -> AsyncThrowingStream<String, Error>
    func stop() async
}
