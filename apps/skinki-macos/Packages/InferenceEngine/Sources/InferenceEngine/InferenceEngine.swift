import Foundation
import SkinkiCore

// NOTE (implementation): import the MLX modules when wiring up real inference:
//   import MLXLLM
//   import MLXLMCommon
//   import MLXEmbedders
//   import MLXHuggingFace
// and load a model with the `#huggingFaceLoadModelContainer` macro, holding the
// resulting `ModelContainer` inside this actor. See ARCHITECTURE.md §4.

/// On-device Gemma 4 engine. An `actor` so the non-Sendable MLX model is
/// accessed serially and never blocks the main thread.
public actor InferenceEngine: LLMService, EmbeddingService {
    private var loadedTier: ModelTier?
    private var idleTask: Task<Void, Never>?

    public init() {}

    // MARK: LLMService

    public func ensureLoaded(tier: ModelTier) async throws {
        guard loadedTier != tier else { return }
        // TODO: download (if needed) + mmap-load weights via mlx-swift-lm.
        loadedTier = tier
        Log.inference.info("Model loaded: \(tier.rawValue, privacy: .public)")
    }

    public func unload() async {
        guard loadedTier != nil else { return }
        // TODO: release the ModelContainer / MLX buffers.
        loadedTier = nil
        Log.inference.info("Model unloaded")
    }

    public func generate(_ request: GenerationRequest) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                // TODO: build a chat session and stream token deltas from MLX.
                continuation.finish(throwing: InferenceError.notImplemented)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: EmbeddingService

    public func embed(_ text: String) async throws -> [Float] {
        // TODO: run EmbeddingGemma via MLXEmbedders and L2-normalize.
        throw InferenceError.notImplemented
    }

    // MARK: Idle management (Pillar 3)

    /// Reset the idle timer; unloads the model after `AppConfig.idleUnloadInterval`.
    public func touchIdleTimer() {
        idleTask?.cancel()
        idleTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(AppConfig.idleUnloadInterval))
            guard !Task.isCancelled else { return }
            await self?.unload()
        }
    }
}

public enum InferenceError: Error, Sendable {
    case notImplemented
    case modelNotLoaded
    case downloadFailed(String)
}
