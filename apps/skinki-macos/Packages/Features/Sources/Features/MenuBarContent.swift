import SwiftUI
import DesignSystem

/// The Status Bar menu content (Skinki's calm home at rest).
public struct MenuBarContent: View {
    private let environment: AppEnvironment
    private let onOpenChat: () -> Void
    private let onOpenSettings: () -> Void

    public init(
        environment: AppEnvironment,
        onOpenChat: @escaping () -> Void,
        onOpenSettings: @escaping () -> Void
    ) {
        self.environment = environment
        self.onOpenChat = onOpenChat
        self.onOpenSettings = onOpenSettings
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            Button("Ask Skinki…", action: onOpenChat)
            Divider()
            Button("Settings…", action: onOpenSettings)
            Button("Quit Skinki") { NSApplication.shared.terminate(nil) }
        }
        .padding(Spacing.sm)
    }
}
