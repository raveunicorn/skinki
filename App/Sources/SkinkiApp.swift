import SwiftUI
import Features
import DesignSystem

// Skinki — the thin app target. All real functionality lives in the local
// Swift packages (see ARCHITECTURE.md). This file only assembles the scenes.

@main
struct SkinkiApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var environment = AppEnvironment()

    var body: some Scene {
        // Status Bar home (the app is an LSUIElement / menu-bar agent).
        MenuBarExtra {
            MenuBarContent(
                environment: environment,
                onOpenChat: { appDelegate.openChat(environment: environment) },
                onOpenSettings: { openSettingsWindow() }
            )
        } label: {
            Text("🦎")
        }
        .menuBarExtraStyle(.menu)

        // Floating chat HUD (also openable via global hotkey).
        Window("Skinki", id: "chat") {
            ChatHUD(
                viewModel: ChatViewModel(
                    llm: environment.inference,
                    model: environment.selectedModel,
                    mascot: environment.mascot
                ),
                mascot: environment.mascot
            )
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)

        Settings {
            SettingsView(environment: environment)
        }
    }

    private func openSettingsWindow() {
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
    }
}
