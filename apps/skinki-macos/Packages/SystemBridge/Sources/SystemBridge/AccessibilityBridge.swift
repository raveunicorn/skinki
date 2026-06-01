import Foundation
import AppKit
import SkinkiCore

// NOTE (implementation): use the AX* API (AXUIElementCopyAttributeValue with
// kAXFocusedUIElementAttribute / kAXSelectedTextAttribute) to read selection,
// and CGEvent to simulate Cmd+C / Cmd+V / typing for in-place replacement.
// Requires the Accessibility permission (requested during onboarding).

public final class AccessibilityBridge: SelectionReading, InputSimulating, @unchecked Sendable {
    public init() {}

    // MARK: SelectionReading

    public func readSelectedText() async -> String? {
        // TODO: read kAXSelectedTextAttribute from the focused element.
        return nil
    }

    // MARK: InputSimulating

    public func replaceSelection(with text: String) async {
        // TODO: set pasteboard + simulate Cmd+V, or post unicode key events.
    }

    public func type(_ text: String) async {
        // TODO: post CGEvent keyboard events for `text`.
    }

    /// Whether the Accessibility permission is currently granted.
    public static var isTrusted: Bool {
        AXIsProcessTrusted()
    }
}
