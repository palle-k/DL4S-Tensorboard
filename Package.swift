// swift-tools-version:6.2

import PackageDescription

let package = Package(
    name: "DL4STensorboard",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "DL4STensorboard", targets: ["DL4STensorboard"]),
    ],
    dependencies: [
        .package(url: "https://github.com/palle-k/DL4S.git", branch: "master"),
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.38.0"),
        .package(url: "https://github.com/twostraws/SwiftGD", from: "2.6.0"),
    ],
    targets: [
        .target(
            name: "DL4STensorboard",
            dependencies: [
                .product(name: "SwiftProtobuf", package: "swift-protobuf"),
                .product(name: "SwiftGD", package: "SwiftGD"),
                .product(name: "DL4S", package: "DL4S"),
            ],
        ),
        .testTarget(name: "DL4STensorboardTests", dependencies: ["DL4STensorboard"]),
    ],
    swiftLanguageModes: [.v6],
)
