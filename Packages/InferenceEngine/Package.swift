// swift-tools-version: 6.0
import PackageDescription

// InferenceEngine — native Gemma 4 inference + embeddings via Apple MLX.
// No Python: everything runs in-process through mlx-swift-lm.
//
// NOTE: dependency versions are pinned to known-good values at scaffold time;
// re-verify with `swift package update` when implementation begins.
let package = Package(
    name: "InferenceEngine",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "InferenceEngine", targets: ["InferenceEngine"]),
    ],
    dependencies: [
        .package(path: "../SkinkiCore"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "InferenceEngine",
            dependencies: [
                "SkinkiCore",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .testTarget(name: "InferenceEngineTests", dependencies: ["InferenceEngine"]),
    ],
    swiftLanguageModes: [.v5]
)
