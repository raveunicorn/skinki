import SwiftUI

/// Renders the lizard mascot. In the MVP this is a placeholder; it will be
/// replaced by a Rive view bound to `MascotController` (see docs/DESIGN.md).
public struct MascotView: View {
    private let controller: MascotController

    public init(controller: MascotController) {
        self.controller = controller
    }

    public var body: some View {
        // TODO: replace with RiveViewModel(.init(fileName: "skinki"))
        //       and bind `controller.mood` / `controller.energy` to inputs.
        ZStack {
            Circle()
                .fill(Palette.accent.opacity(0.18))
            Text("🦎")
                .font(.system(size: 40))
                .scaleEffect(1 + controller.energy * 0.08)
                .animation(Motion.playful, value: controller.energy)
        }
        .frame(width: 72, height: 72)
        .accessibilityHidden(true)
    }
}
