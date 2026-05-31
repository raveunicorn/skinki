import Foundation
import SkinkiCore

// NOTE (implementation): `import SQLiteVec`, open the database at
// `AppConfig.memoryDatabaseURL`, run `Schema.statements`, and implement the
// write/read paths from docs/MEMORY.md. Embeddings come from the injected
// `EmbeddingService` so this package never depends on the inference backend.

/// Local long-term memory & RAG store. An `actor` to serialize SQLite access.
public actor MemoryStore: MemoryStoring {
    private let embedder: EmbeddingService
    private var isOpen = false

    public init(embedder: EmbeddingService) {
        self.embedder = embedder
    }

    /// Open the database and ensure the schema exists.
    public func open() async throws {
        guard !isOpen else { return }
        try FileManager.default.createDirectory(
            at: AppConfig.supportDirectory, withIntermediateDirectories: true
        )
        // TODO: open SQLite at AppConfig.memoryDatabaseURL + execute Schema.statements.
        isOpen = true
        Log.memory.info("Memory store opened")
    }

    // MARK: MemoryStoring

    public func remember(_ memory: NewMemory) async throws {
        // TODO: dedup via embedding search, then insert into memories + vec_memories.
        _ = try await embedder.embed(memory.content)
        throw MemoryError.notImplemented
    }

    public func retrieve(for query: String, limit: Int) async throws -> [RetrievedMemory] {
        // TODO: embed query, KNN over vec_memories, re-rank, return top results.
        _ = try await embedder.embed(query)
        return []
    }

    public func pinned() async throws -> [Memory] { [] }

    public func forget(id: Memory.ID) async throws {
        // TODO: delete from memories + vec_memories.
        throw MemoryError.notImplemented
    }

    public func forgetAll() async throws {
        // TODO: truncate all tables.
        throw MemoryError.notImplemented
    }
}

public enum MemoryError: Error, Sendable {
    case notImplemented
    case notOpen
}
