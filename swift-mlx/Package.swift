// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "OPFMLXDaemon",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "opf-mlx-daemon", targets: ["OPFMLXDaemon"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0")
    ],
    targets: [
        .executableTarget(
            name: "OPFMLXDaemon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers")
            ],
            path: "Sources/OPFMLXDaemon"
        )
    ]
)
