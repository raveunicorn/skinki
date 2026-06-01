import SwiftUI
import Observation

// NOTE (implementation): back this with a Rive `RiveViewModel` / state machine.
// `MascotMood` and the trigger methods map to Rive state-machine inputs so the
// rest of the app drives the lizard at the level of intent, never Rive directly.
// See docs/DESIGN.md §4.

/// The lizard's emotional state.
public enum MascotMood: String, Sendable, CaseIterable {
    case idle, curious, thinking, talking, happy, confused, error, sleeping
}

/// Drives the mascot at the level of intent. Inject into views; call high-level
/// methods in response to app events.
@Observable
@MainActor
public final class MascotController {
    public private(set) var mood: MascotMood = .idle
    public private(set) var energy: Double = 0.3

    public init() {}

    public func set(_ mood: MascotMood, energy: Double? = nil) {
        withAnimation(Motion.playful) {
            self.mood = mood
            if let energy { self.energy = energy }
        }
    }

    public func greet() { set(.curious, energy: 0.7) }
    public func think() { set(.thinking, energy: 0.5) }
    public func talk() { set(.talking, energy: 0.8) }
    public func celebrate() { set(.happy, energy: 1.0) }
    public func confused() { set(.confused, energy: 0.4) }
    public func fail() { set(.error, energy: 0.4) }
    public func rest() { set(.idle, energy: 0.3) }
}
