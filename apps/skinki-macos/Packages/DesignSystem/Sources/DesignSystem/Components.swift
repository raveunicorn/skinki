import SwiftUI

/// A blurred, elevated surface using a native macOS material.
public struct GlassPanel<Content: View>: View {
    private let content: Content
    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }
    public var body: some View {
        content
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: Radius.lg, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Radius.lg, style: .continuous)
                    .strokeBorder(.white.opacity(0.08), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.15), radius: 20, y: 8)
    }
}

/// Primary joy-design button with a press spring + haptic.
public struct JoyButton: View {
    private let title: String
    private let action: () -> Void
    @State private var pressed = false

    public init(_ title: String, action: @escaping () -> Void) {
        self.title = title
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Text(title)
                .font(TypeScale.headline)
                .padding(.horizontal, Spacing.lg)
                .padding(.vertical, Spacing.sm)
        }
        .buttonStyle(.plain)
        .background(Palette.accent, in: RoundedRectangle(cornerRadius: Radius.md, style: .continuous))
        .foregroundStyle(.white)
        .scaleEffect(pressed ? 0.96 : 1)
        .animation(Motion.snappy, value: pressed)
    }
}
