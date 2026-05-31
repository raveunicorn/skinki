import Foundation

/// SQL schema for the local memory database. See docs/MEMORY.md.
enum Schema {
    /// EmbeddingGemma output dimension — confirm against the loaded model.
    static let embeddingDimension = 768

    static let statements: [String] = [
        """
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY,
            kind        TEXT NOT NULL,
            content     TEXT NOT NULL,
            source      TEXT,
            importance  REAL DEFAULT 0.5,
            created_at  INTEGER NOT NULL,
            last_used   INTEGER,
            use_count   INTEGER DEFAULT 0,
            metadata    TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id          INTEGER PRIMARY KEY,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            session_id  TEXT,
            created_at  INTEGER NOT NULL
        );
        """,
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            memory_id   INTEGER PRIMARY KEY,
            embedding   FLOAT[\(embeddingDimension)]
        );
        """,
    ]
}
