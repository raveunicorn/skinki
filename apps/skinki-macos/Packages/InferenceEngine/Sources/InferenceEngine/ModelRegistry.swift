import Foundation
import SkinkiCore

/// Maps Skinki's `ModelTier` to concrete Hugging Face model identifiers.
///
/// At implementation time these resolve to `LLMRegistry` configurations in
/// `mlx-swift-lm` (e.g. `#huggingFaceLoadModelContainer(configuration:)`).
public struct ModelDescriptor: Sendable {
    public let tier: ModelTier
    public let huggingFaceID: String
    public let displayName: String
    public let approxDownloadGB: Double
}

public enum ModelRegistry {
    public static func descriptor(for tier: ModelTier) -> ModelDescriptor {
        switch tier {
        case .e2b:
            return .init(tier: tier, huggingFaceID: "google/gemma-4-e2b-it",
                         displayName: "Gemma 4 E2B", approxDownloadGB: 4)
        case .e4b:
            return .init(tier: tier, huggingFaceID: "google/gemma-4-e4b-it",
                         displayName: "Gemma 4 E4B", approxDownloadGB: 6)
        case .moe26b:
            return .init(tier: tier, huggingFaceID: "google/gemma-4-26b-a4b-it",
                         displayName: "Gemma 4 26B-A4B", approxDownloadGB: 18)
        }
    }

    /// Embedding model used for RAG (see docs/MEMORY.md).
    public static let embeddingModelID = "google/embeddinggemma-300m"
}
