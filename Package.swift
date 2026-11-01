// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "vector",
    platforms: [.macOS(.v11), .iOS(.v12)],
    products: [
        .library(
            name: "vector",
            targets: ["vector"])
    ],
    targets: [
        .binaryTarget(
            name: "vectorBinary",
            url: "https://github.com/sqliteai/sqlite-vector/releases/download/1.2.0/vector-apple-xcframework-1.2.0.zip",
            checksum: "b6b1a18078324a1d6aae2b56e1b5710946a094dab8b588325236a495324cc06e"
        ),
        .target(
            name: "vector",
            dependencies: ["vectorBinary"],
            path: "packages/swift"
        ),
    ]
)
