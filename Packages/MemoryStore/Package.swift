// swift-tools-version: 6.0
import PackageDescription

// MemoryStore — long-term memory & RAG over SQLite + sqlite-vec.
// See docs/MEMORY.md for the schema and read/write paths.
//
// NOTE: `SQLiteVec` is a Swift wrapper around the sqlite-vec extension; verify
// the package/product name and version when implementation begins.
let package = Package(
    name: "MemoryStore",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MemoryStore", targets: ["MemoryStore"]),
    ],
    dependencies: [
        .package(path: "../SkinkiCore"),
        .package(url: "https://github.com/jkrukowski/SQLiteVec", from: "0.0.10"),
    ],
    targets: [
        .target(
            name: "MemoryStore",
            dependencies: [
                "SkinkiCore",
                .product(name: "SQLiteVec", package: "SQLiteVec"),
            ]
        ),
        .testTarget(name: "MemoryStoreTests", dependencies: ["MemoryStore"]),
    ],
    swiftLanguageModes: [.v5]
)
