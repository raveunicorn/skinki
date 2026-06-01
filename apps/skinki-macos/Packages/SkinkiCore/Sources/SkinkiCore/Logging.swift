import Foundation
import OSLog

/// Centralized logging. Use `Log.<area>` instead of scattering `Logger` instances.
public enum Log {
    private static let subsystem = "com.skinki.app"

    public static let app = Logger(subsystem: subsystem, category: "app")
    public static let inference = Logger(subsystem: subsystem, category: "inference")
    public static let memory = Logger(subsystem: subsystem, category: "memory")
    public static let system = Logger(subsystem: subsystem, category: "system")
    public static let voice = Logger(subsystem: subsystem, category: "voice")
    public static let ui = Logger(subsystem: subsystem, category: "ui")
}
