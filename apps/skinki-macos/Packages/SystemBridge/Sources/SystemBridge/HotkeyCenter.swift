import Foundation
import SkinkiCore

// NOTE (implementation): register global hotkeys via the Carbon
// `RegisterEventHotKey` API (still the standard for system-wide shortcuts),
// or `NSEvent.addGlobalMonitorForEvents` for monitoring.

public final class HotkeyCenter: HotkeyRegistering, @unchecked Sendable {
    private var handlers: [String: @Sendable () -> Void] = [:]

    public init() {}

    public func register(_ hotkey: Hotkey, id: String, handler: @escaping @Sendable () -> Void) throws {
        handlers[id] = handler
        // TODO: RegisterEventHotKey + dispatch to `handler` on press.
        Log.system.info("Hotkey registered: \(id, privacy: .public)")
    }

    public func unregister(id: String) {
        handlers[id] = nil
        // TODO: UnregisterEventHotKey.
    }
}
