// Copyright (c) 2025 SQLite Cloud, Inc.
// Licensed under the Apache License 2.0 (see LICENSE.md).

import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as p;

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;

    final codeConfig = input.config.code;
    final os = codeConfig.targetOS;
    final arch = codeConfig.targetArchitecture;

    final binaryPath = _resolveBinaryPath(os, arch, codeConfig);
    if (binaryPath == null) {
      throw UnsupportedError('sqlite_vector does not support $os $arch.');
    }

    final nativeLibDir = p.join(
      input.packageRoot.toFilePath(),
      'native_libraries',
    );
    final file = File(p.join(nativeLibDir, binaryPath));
    if (!file.existsSync()) {
      throw StateError(
        'Pre-built binary not found: ${file.path}. '
        'Run the CI pipeline to populate native_libraries/.',
      );
    }

    output.dependencies.add(file.uri);

    final assetFile = await _prepareAssetFile(
      input: input,
      os: os,
      arch: arch,
      config: codeConfig,
      file: file,
    );

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'src/native/sqlite_vector_extension.dart',
        linkMode: DynamicLoadingBundled(),
        file: assetFile.uri,
      ),
    );
  });
}

Future<File> _prepareAssetFile({
  required BuildInput input,
  required OS os,
  required Architecture arch,
  required CodeConfig config,
  required File file,
}) async {
  if (os != OS.iOS || config.iOS.targetSdk == IOSSdk.iPhoneOS) {
    return file;
  }

  final thinArch = switch (arch) {
    Architecture.arm64 => 'arm64',
    Architecture.x64 => 'x86_64',
    _ => null,
  };
  if (thinArch == null) {
    return file;
  }

  final outputName = 'vector_ios_sim_$thinArch.dylib';
  final outputFile = File.fromUri(input.outputDirectory.resolve(outputName));
  await outputFile.parent.create(recursive: true);

  final result = await Process.run('/usr/bin/lipo', [
    file.path,
    '-thin',
    thinArch,
    '-output',
    outputFile.path,
  ]);
  if (result.exitCode != 0) {
    throw StateError(
      'Failed to thin sqlite_vector iOS simulator binary for $thinArch: '
      '${result.stderr}',
    );
  }

  return outputFile;
}

String? _resolveBinaryPath(OS os, Architecture arch, CodeConfig config) {
  if (os == OS.android) {
    return switch (arch) {
      Architecture.arm64 => 'android/vector_android_arm64.so',
      Architecture.arm => 'android/vector_android_arm.so',
      Architecture.x64 => 'android/vector_android_x64.so',
      _ => null,
    };
  }

  if (os == OS.iOS) {
    final sdk = config.iOS.targetSdk;
    if (sdk == IOSSdk.iPhoneOS) {
      return 'ios/vector_ios_arm64.dylib';
    }
    // Simulator: fat binary (arm64 + x64)
    return 'ios-sim/vector_ios-sim.dylib';
  }

  if (os == OS.macOS) {
    return switch (arch) {
      Architecture.arm64 => 'mac/vector_mac_arm64.dylib',
      Architecture.x64 => 'mac/vector_mac_x64.dylib',
      _ => null,
    };
  }

  if (os == OS.linux) {
    return switch (arch) {
      Architecture.x64 => 'linux/vector_linux_x64.so',
      Architecture.arm64 => 'linux/vector_linux_arm64.so',
      _ => null,
    };
  }

  if (os == OS.windows) {
    return switch (arch) {
      Architecture.x64 => 'windows/vector_windows_x64.dll',
      _ => null,
    };
  }

  return null;
}
