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
            url: "https://github.com/sqliteai/sqlite-vector/releases/download/1.0.0/vector-apple-xcframework-1.0.0.zip",
            checksum: "26962a269a0e7f5da3fca421c7c43a9d7542367ccadb898f8e4fc057ad18a18c"
        ),
        .target(
            name: "vector",
            dependencies: ["vectorBinary"],
            path: "packages/swift"
        ),
    ]
)
