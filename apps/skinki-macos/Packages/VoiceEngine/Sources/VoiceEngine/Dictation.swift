import Foundation
import SkinkiCore

// NOTE (implementation): use the Speech framework (`SFSpeechRecognizer` +
// `SFSpeechAudioBufferRecognitionRequest` with AVAudioEngine) for on-device
// transcription. Requires Microphone + Speech Recognition permissions.

/// On-device speech-to-text dictation.
public final class Dictation: DictationService, @unchecked Sendable {
    public init() {}

    public func transcribe() -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            // TODO: start AVAudioEngine + SFSpeechRecognizer, yield partial results.
            continuation.finish(throwing: VoiceError.notImplemented)
        }
    }

    public func stop() async {
        // TODO: stop the audio engine + recognition task.
    }
}

public enum VoiceError: Error, Sendable {
    case notImplemented
    case permissionDenied
}
