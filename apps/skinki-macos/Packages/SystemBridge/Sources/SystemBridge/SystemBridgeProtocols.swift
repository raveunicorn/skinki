import Foundation

// Protocol surface for the system layer. Feature code depends on these, so the
// risky concrete implementations stay mockable and isolated.

/// Reads the user's current text selection in the frontmost app (Accessibility).
public protocol SelectionReading: Sendable {
    func readSelectedText() async -> String?
}

/// Simulates keyboard/paste input to replace text in place (CGEvent).
public protocol InputSimulating: Sendable {
    func replaceSelection(with text: String) async
    func type(_ text: String) async
}

public struct Hotkey: Sendable, Hashable {
    public var keyCode: UInt16
    public var modifiers: UInt   // Carbon modifier flags
    public init(keyCode: UInt16, modifiers: UInt) {
        self.keyCode = keyCode
        self.modifiers = modifiers
    }
}

/// Registers global hotkeys that fire regardless of the focused app.
public protocol HotkeyRegistering: Sendable {
    func register(_ hotkey: Hotkey, id: String, handler: @escaping @Sendable () -> Void) throws
    func unregister(id: String)
}

/// Permission checks/prompts (TCC).
public enum SystemPermission: Sendable {
    case accessibility
    case microphone
    case speechRecognition
    case screenRecording
    case fullDiskAccess
}

public protocol PermissionChecking: Sendable {
    func status(for permission: SystemPermission) async -> Bool
    func requestAccess(for permission: SystemPermission) async
}
