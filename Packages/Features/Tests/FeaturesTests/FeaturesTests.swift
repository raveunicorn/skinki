import Testing
import SkinkiCore
@testable import Features

@MainActor
@Test func appEnvironmentSelectsAModelForHardware() {
    let env = AppEnvironment()
    #expect(ModelTier.allCases.contains(env.selectedModel))
}
