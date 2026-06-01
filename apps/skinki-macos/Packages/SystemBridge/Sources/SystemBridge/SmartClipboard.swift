import Foundation
import AppKit
import SkinkiCore

/// Clipboard history with quick recall (the "Smart Clipboard" feature).
public actor SmartClipboard {
    public struct Entry: Identifiable, Sendable {
        public let id = UUID()
        public let text: String
        public let date: Date
    }

    private var history: [Entry] = []
    private let maxEntries: Int

    public init(maxEntries: Int = 50) {
        self.maxEntries = maxEntries
    }

    public func snapshot() -> [Entry] { history }

    /// Poll the system pasteboard and record new text values.
    public func capture() {
        // TODO: observe NSPasteboard.changeCount and append new string contents.
    }
}
