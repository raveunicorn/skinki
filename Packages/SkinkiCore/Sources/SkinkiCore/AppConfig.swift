import Foundation

/// Static configuration and well-known paths.
public enum AppConfig {
    public static let bundleIdentifier = "com.skinki.app"

    /// `~/Library/Application Support/Skinki`
    public static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("Skinki", isDirectory: true)
    }

    /// Local memory database (see docs/MEMORY.md).
    public static var memoryDatabaseURL: URL {
        supportDirectory.appendingPathComponent("memory.sqlite")
    }

    /// Where downloaded model weights are cached.
    public static var modelsDirectory: URL {
        supportDirectory.appendingPathComponent("Models", isDirectory: true)
    }

    /// Idle interval after which the model is unloaded (Pillar 3).
    public static let idleUnloadInterval: TimeInterval = 90
}
