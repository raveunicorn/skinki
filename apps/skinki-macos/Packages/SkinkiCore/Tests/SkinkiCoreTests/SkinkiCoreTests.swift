import Testing
@testable import SkinkiCore

@Test func recommendedModelForHardwareTier() {
    #expect(ModelTier.recommended(for: .light) == .e2b)
    #expect(ModelTier.recommended(for: .standard) == .e4b)
    #expect(ModelTier.recommended(for: .pro) == .moe26b)
}
