import AppKit
import SwiftUI
import SkinkiCore
import SystemBridge
import Features

/// App-level lifecycle: registers the global hotkey and routes to the chat HUD.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        Log.app.info("Skinki launched")
        // TODO: register the global summon hotkey via SystemBridge.HotkeyCenter
        //       and show onboarding on first run (Accessibility permission).
    }

    @MainActor
    func openChat(environment: AppEnvironment) {
        // TODO: present the ChatHUD as a borderless floating NSPanel near the
        //       cursor instead of a standard window (see docs/DESIGN.md §5).
        NSApp.activate(ignoringOtherApps: true)
    }
}
