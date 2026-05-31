import Testing
import SkinkiCore
@testable import InferenceEngine

@Test func hardwareDetectionReturnsATier() {
    let tier = HardwareDetector.detect()
    #expect(HardwareTier.allCases.contains(tier))
}

@Test func registryHasIdentifierForEveryTier() {
    for tier in ModelTier.allCases {
        #expect(!ModelRegistry.descriptor(for: tier).huggingFaceID.isEmpty)
    }
}
