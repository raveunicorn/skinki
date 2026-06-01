import SwiftUI
import DesignSystem

/// Friendly, zero-friction onboarding where the mascot guides the user through
/// granting permissions one contextual step at a time. See docs/DESIGN.md §5.
public struct OnboardingView: View {
    private let mascot: MascotController
    private let onFinish: () -> Void

    public init(mascot: MascotController, onFinish: @escaping () -> Void) {
        self.mascot = mascot
        self.onFinish = onFinish
    }

    public var body: some View {
        VStack(spacing: Spacing.xl) {
            MascotView(controller: mascot)
                .frame(width: 120, height: 120)
            Text("Hi, I'm Skinki")
                .font(TypeScale.largeTitle)
            Text("Your private, on-device assistant. Let's set up a couple of things together.")
                .font(TypeScale.body)
                .foregroundStyle(Palette.textSecondary)
                .multilineTextAlignment(.center)
            // TODO: step-by-step permission requests (Accessibility, Microphone, …).
            JoyButton("Let's go", action: onFinish)
        }
        .padding(Spacing.xxl)
        .frame(width: 520, height: 480)
        .onAppear { mascot.greet() }
    }
}
