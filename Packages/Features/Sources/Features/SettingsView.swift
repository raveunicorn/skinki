import SwiftUI
import SkinkiCore
import DesignSystem

/// Native settings: model tier, voice, hotkeys, memory, privacy.
public struct SettingsView: View {
    @State private var environment: AppEnvironment

    public init(environment: AppEnvironment) {
        _environment = State(initialValue: environment)
    }

    public var body: some View {
        Form {
            Section("Model") {
                Picker("Model tier", selection: $environment.selectedModel) {
                    ForEach(ModelTier.allCases) { tier in
                        Text(tier.rawValue.uppercased()).tag(tier)
                    }
                }
                LabeledContent("Detected hardware", value: environment.hardwareTier.rawValue.capitalized)
            }
            Section("Privacy") {
                Text("Everything runs on your Mac. Nothing is sent to the cloud.")
                    .font(TypeScale.caption)
                    .foregroundStyle(Palette.textSecondary)
            }
            // TODO: voice, hotkeys, and memory controls.
        }
        .formStyle(.grouped)
        .frame(width: 480, height: 360)
    }
}
