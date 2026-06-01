import SwiftUI

// Design tokens — see docs/DESIGN.md. Nothing in the app should hard-code
// colors, spacing, radii, or animation timings; reference these instead.

public enum Spacing {
    public static let xs: CGFloat = 4
    public static let sm: CGFloat = 8
    public static let md: CGFloat = 12
    public static let lg: CGFloat = 16
    public static let xl: CGFloat = 24
    public static let xxl: CGFloat = 32
}

public enum Radius {
    public static let sm: CGFloat = 8
    public static let md: CGFloat = 14
    public static let lg: CGFloat = 22
    public static let pill: CGFloat = 999
}

public enum Palette {
    public static let accent = Color("AccentColor")
    public static let surface = Color(nsColor: .windowBackgroundColor)
    public static let textPrimary = Color(nsColor: .labelColor)
    public static let textSecondary = Color(nsColor: .secondaryLabelColor)
    public static let success = Color.green
    public static let warning = Color.orange
    public static let danger = Color.red
}

public enum Motion {
    /// Quick, responsive interactions.
    public static let snappy = Animation.spring(response: 0.30, dampingFraction: 0.85)
    /// Smooth, content-level transitions.
    public static let smooth = Animation.spring(response: 0.42, dampingFraction: 0.90)
    /// Playful, slightly bouncy accents (mascot, celebrations).
    public static let playful = Animation.spring(response: 0.36, dampingFraction: 0.62)

    public static let fast: Double = 0.12
    public static let base: Double = 0.20
    public static let slow: Double = 0.35
}

public enum TypeScale {
    public static let largeTitle = Font.system(.largeTitle, design: .rounded, weight: .bold)
    public static let title = Font.system(.title2, design: .rounded, weight: .semibold)
    public static let headline = Font.system(.headline, design: .rounded)
    public static let body = Font.system(.body)
    public static let callout = Font.system(.callout)
    public static let caption = Font.system(.caption)
}
