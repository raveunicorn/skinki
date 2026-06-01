import Testing
@testable import DesignSystem

@MainActor
@Test func mascotControllerUpdatesMood() {
    let controller = MascotController()
    controller.think()
    #expect(controller.mood == .thinking)
    controller.celebrate()
    #expect(controller.mood == .happy)
}
