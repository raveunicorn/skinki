import Testing
@testable import VoiceEngine

@Test func synthesizerInitializes() {
    _ = SystemSpeechSynthesizer()
    #expect(Bool(true))
}
