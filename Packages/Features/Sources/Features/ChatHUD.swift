import SwiftUI
import SkinkiCore
import DesignSystem

/// The floating, spotlight-style chat HUD summoned by a global hotkey.
/// See docs/DESIGN.md §5.
public struct ChatHUD: View {
    @State private var model: ChatViewModel
    private let mascot: MascotController

    public init(viewModel: ChatViewModel, mascot: MascotController) {
        _model = State(initialValue: viewModel)
        self.mascot = mascot
    }

    public var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: Spacing.md) {
                header
                transcript
                inputBar
            }
            .padding(Spacing.lg)
        }
        .frame(width: 560)
        .onAppear { mascot.greet() }
    }

    private var header: some View {
        HStack(spacing: Spacing.sm) {
            MascotView(controller: mascot)
                .frame(width: 40, height: 40)
            Text("Skinki")
                .font(TypeScale.title)
            Spacer()
        }
    }

    private var transcript: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.sm) {
                ForEach(model.messages) { message in
                    Text(message.content)
                        .font(TypeScale.body)
                        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
                        .foregroundStyle(message.role == .user ? Palette.textPrimary : Palette.textSecondary)
                }
            }
        }
        .frame(maxHeight: 320)
    }

    private var inputBar: some View {
        HStack(spacing: Spacing.sm) {
            TextField("Ask Skinki…", text: $model.input)
                .textFieldStyle(.plain)
                .font(TypeScale.body)
                .onSubmit { model.send() }
            JoyButton(model.isGenerating ? "Stop" : "Send") {
                model.isGenerating ? model.cancel() : model.send()
            }
        }
        .padding(Spacing.sm)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: Radius.md, style: .continuous))
    }
}
