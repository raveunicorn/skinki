import Foundation

// MARK: - Models

/// Hardware capability tier, derived from unified memory (see `HardwareTier.detect`).
public enum HardwareTier: String, Sendable, CaseIterable {
    case light      // < 16 GB
    case standard   // 16–32 GB
    case pro        // >= 32 GB
}

/// A concrete Gemma 4 model variant Skinki can run.
public enum ModelTier: String, Sendable, CaseIterable, Identifiable {
    case e2b        // Gemma 4 E2B — fallback for constrained machines
    case e4b        // Gemma 4 E4B — base Macs (16 GB)
    case moe26b     // Gemma 4 26B-A4B MoE — Mac Studio / 32 GB+

    public var id: String { rawValue }

    /// Recommended default model for a given hardware tier.
    public static func recommended(for hardware: HardwareTier) -> ModelTier {
        switch hardware {
        case .light: return .e2b
        case .standard: return .e4b
        case .pro: return .moe26b
        }
    }
}

public enum ChatRole: String, Sendable, Codable {
    case system, user, assistant
}

public struct ChatMessage: Identifiable, Sendable, Codable, Equatable {
    public let id: UUID
    public var role: ChatRole
    public var content: String
    public var createdAt: Date

    public init(id: UUID = UUID(), role: ChatRole, content: String, createdAt: Date = .now) {
        self.id = id
        self.role = role
        self.content = content
        self.createdAt = createdAt
    }
}

/// A request to the LLM. Kept backend-agnostic.
public struct GenerationRequest: Sendable {
    public var system: String?
    public var messages: [ChatMessage]
    public var maxTokens: Int
    public var temperature: Double

    public init(
        system: String? = nil,
        messages: [ChatMessage],
        maxTokens: Int = 1024,
        temperature: Double = 0.7
    ) {
        self.system = system
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
    }
}

// MARK: - Memory models (see docs/MEMORY.md)

public enum MemoryKind: String, Sendable, Codable {
    case fact, preference, tone, path, codeStyle = "code_style", snippet
}

public struct NewMemory: Sendable {
    public var kind: MemoryKind
    public var content: String
    public var source: String?
    public var importance: Double

    public init(kind: MemoryKind, content: String, source: String? = nil, importance: Double = 0.5) {
        self.kind = kind
        self.content = content
        self.source = source
        self.importance = importance
    }
}

public struct Memory: Identifiable, Sendable, Equatable {
    public let id: Int64
    public var kind: MemoryKind
    public var content: String
    public var importance: Double
    public var createdAt: Date

    public init(id: Int64, kind: MemoryKind, content: String, importance: Double, createdAt: Date) {
        self.id = id
        self.kind = kind
        self.content = content
        self.importance = importance
        self.createdAt = createdAt
    }
}

public struct RetrievedMemory: Sendable, Identifiable {
    public var id: Int64 { memory.id }
    public var memory: Memory
    public var score: Double

    public init(memory: Memory, score: Double) {
        self.memory = memory
        self.score = score
    }
}
