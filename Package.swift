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
            url: "https://github.com/sqliteai/sqlite-vector/releases/download/1.1.0/vector-apple-xcframework-1.1.0.zip",
            checksum: "a9fc6606d86460d8bbfd946e257e2bc4f861a99525f584edf08e1056d71213e4"
        ),
        .target(
            name: "vector",
            dependencies: ["vectorBinary"],
            path: "packages/swift"
        ),
    ]
)
