// swift-tools-version:5.2

import PackageDescription

let package = Package(
    name: "DL4STensorboard",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15),
        .watchOS(.v6),
        .tvOS(.v13)
    ],
    products: [
        .library(name: "DL4STensorboard", targets: ["DL4STensorboard"]),
    ],
    dependencies: [
        // .package(url: "https://github.com/palle-k/DL4S.git", .branch("master")),
        .package(name: "SwiftProtobuf", url: "https://github.com/apple/swift-protobuf.git", from: "1.6.0"),
        .package(url: "https://github.com/twostraws/SwiftGD", from: "2.0.0")
    ],
    targets: [
        .target(name: "DL4STensorboard", dependencies: ["SwiftProtobuf", "SwiftGD"]),
        .testTarget(name: "DL4STensorboardTests", dependencies: ["DL4STensorboard"])
    ]
)
