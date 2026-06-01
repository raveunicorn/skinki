import Foundation
import SkinkiCore

public enum HardwareDetector {
    /// Total unified memory in bytes.
    public static var physicalMemory: UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    /// Classify the machine into a `HardwareTier` from unified memory.
    public static func detect() -> HardwareTier {
        let gb = Double(physicalMemory) / 1_073_741_824.0
        switch gb {
        case ..<16: return .light
        case 16..<32: return .standard
        default: return .pro
        }
    }
}
