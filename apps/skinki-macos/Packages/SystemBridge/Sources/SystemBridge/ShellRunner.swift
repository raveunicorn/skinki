import Foundation
import SkinkiCore

/// Runs shell commands with mandatory human-in-the-loop confirmation for any
/// destructive action (Pillar: safety). See ARCHITECTURE.md §6.
public actor ShellRunner {
    public struct Command: Sendable {
        public let executable: String
        public let arguments: [String]
        /// Destructive commands MUST be confirmed by the user before running.
        public let isDestructive: Bool
        public init(executable: String, arguments: [String], isDestructive: Bool) {
            self.executable = executable
            self.arguments = arguments
            self.isDestructive = isDestructive
        }
    }

    public struct Result: Sendable {
        public let stdout: String
        public let stderr: String
        public let exitCode: Int32
    }

    /// Confirmation gate the UI must satisfy for destructive commands.
    public typealias ConfirmationHandler = @Sendable (Command) async -> Bool

    private let confirm: ConfirmationHandler

    public init(confirm: @escaping ConfirmationHandler) {
        self.confirm = confirm
    }

    public func run(_ command: Command) async throws -> Result {
        if command.isDestructive {
            guard await confirm(command) else { throw ShellError.cancelledByUser }
        }
        // TODO: execute via Process, capture stdout/stderr.
        throw ShellError.notImplemented
    }
}

public enum ShellError: Error, Sendable {
    case notImplemented
    case cancelledByUser
}
