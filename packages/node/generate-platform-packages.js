#!/usr/bin/env node

/**
 * Generates platform-specific packages dynamically
 *
 * This script creates npm packages for each platform from templates,
 * eliminating the need to maintain nearly-identical files in the repo.
 *
 * Usage:
 *   node generate-platform-packages.js <version> <artifacts-dir> <output-dir>
 *
 * Example:
 *   node generate-platform-packages.js 0.9.45 ./artifacts ./platform-packages
 */

const fs = require('fs');
const path = require('path');

// Platform configuration
const PLATFORMS = [
  {
    name: 'darwin-arm64',
    os: ['darwin'],
    cpu: ['arm64'],
    description: 'SQLite Vector extension for macOS ARM64 (Apple Silicon)',
    binaryName: 'vector.dylib',
    artifactFolder: 'vector-macos-arm64',
  },
  {
    name: 'darwin-x86_64',
    os: ['darwin'],
    cpu: ['x64', 'ia32'],
    description: 'SQLite Vector extension for macOS x86_64 (Intel)',
    binaryName: 'vector.dylib',
    artifactFolder: 'vector-macos-x86_64',
  },
  {
    name: 'linux-arm64',
    os: ['linux'],
    cpu: ['arm64'],
    description: 'SQLite Vector extension for Linux ARM64 (glibc)',
    binaryName: 'vector.so',
    artifactFolder: 'vector-linux-arm64',
  },
  {
    name: 'linux-arm64-musl',
    os: ['linux'],
    cpu: ['arm64'],
    description: 'SQLite Vector extension for Linux ARM64 (musl)',
    binaryName: 'vector.so',
    artifactFolder: 'vector-linux-musl-arm64',
  },
  {
    name: 'linux-x86_64',
    os: ['linux'],
    cpu: ['x64', 'ia32'],
    description: 'SQLite Vector extension for Linux x86_64 (glibc)',
    binaryName: 'vector.so',
    artifactFolder: 'vector-linux-x86_64',
  },
  {
    name: 'linux-x86_64-musl',
    os: ['linux'],
    cpu: ['x64', 'ia32'],
    description: 'SQLite Vector extension for Linux x86_64 (musl)',
    binaryName: 'vector.so',
    artifactFolder: 'vector-linux-musl-x86_64',
  },
  {
    name: 'win32-x86_64',
    os: ['win32'],
    cpu: ['x64', 'ia32'],
    description: 'SQLite Vector extension for Windows x86_64',
    binaryName: 'vector.dll',
    artifactFolder: 'vector-windows-x86_64',
  },
];

/**
 * Generate package.json for a platform
 */
function generatePackageJson(platform, version) {
  return {
    name: `@sqliteai/sqlite-vector-${platform.name}`,
    version: version,
    description: platform.description,
    main: 'index.js',
    os: platform.os,
    cpu: platform.cpu,
    files: [
      platform.binaryName,
      'index.js',
      'README.md',
      'LICENSE.md',
    ],
    keywords: [
      'sqlite',
      'vector',
      ...platform.name.split('-'),
    ],
    author: 'Gioele Cantoni (gioele@sqlitecloud.io)',
    license: 'Apache-2.0',
    repository: {
      type: 'git',
      url: 'https://github.com/sqliteai/sqlite-vector.git',
      directory: 'packages/node',
    },
    engines: {
      node: '>=16.0.0',
    },
  };
}

/**
 * Generate index.js for a platform
 */
function generateIndexJs(platform) {
  return `const { join } = require('path');

module.exports = {
  path: join(__dirname, '${platform.binaryName}')
};
`;
}

/**
 * Generate README.md for a platform
 */
function generateReadme(platform, version) {
  return `# @sqliteai/sqlite-vector-${platform.name}

${platform.description}

**Version:** ${version}

This is a platform-specific package for [@sqliteai/sqlite-vector](https://www.npmjs.com/package/@sqliteai/sqlite-vector).

It is installed automatically as an optional dependency and should not be installed directly.

## Installation

Install the main package instead:

\`\`\`bash
npm install @sqliteai/sqlite-vector
\`\`\`

## Platform

- **OS:** ${platform.os.join(', ')}
- **CPU:** ${platform.cpu.join(', ')}
- **Binary:** ${platform.binaryName}

## License

See [LICENSE.md](./LICENSE.md) in the root directory.
`;
}

/**
 * Main function
 */
function main() {
  const args = process.argv.slice(2);

  if (args.length < 3) {
    console.error('Usage: node generate-platform-packages.js <version> <artifacts-dir> <output-dir>');
    console.error('Example: node generate-platform-packages.js 0.9.45 ./artifacts ./platform-packages');
    process.exit(1);
  }

  const [version, artifactsDir, outputDir] = args;

  // Find LICENSE.md (should be in repo root)
  const licensePath = path.resolve(__dirname, '../../LICENSE.md');
  if (!fs.existsSync(licensePath)) {
    console.error(`Error: LICENSE.md not found at ${licensePath}`);
    process.exit(1);
  }

  // Validate version format
  if (!/^\d+\.\d+\.\d+$/.test(version)) {
    console.error(`Error: Invalid version format: ${version}`);
    console.error('Version must be in semver format (e.g., 0.9.45)');
    process.exit(1);
  }

  console.log(`Generating platform packages version ${version}...\n`);

  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  let successCount = 0;
  let errorCount = 0;

  // Generate each platform package
  for (const platform of PLATFORMS) {
    const platformDir = path.join(outputDir, platform.name);
    const artifactPath = path.join(artifactsDir, platform.artifactFolder, platform.binaryName);

    try {
      // Create platform directory
      fs.mkdirSync(platformDir, { recursive: true });

      // Generate package.json
      const packageJson = generatePackageJson(platform, version);
      fs.writeFileSync(
        path.join(platformDir, 'package.json'),
        JSON.stringify(packageJson, null, 2) + '\n'
      );

      // Generate index.js
      const indexJs = generateIndexJs(platform);
      fs.writeFileSync(path.join(platformDir, 'index.js'), indexJs);

      // Generate README.md
      const readme = generateReadme(platform, version);
      fs.writeFileSync(path.join(platformDir, 'README.md'), readme);

      // Copy LICENSE.md
      fs.copyFileSync(licensePath, path.join(platformDir, 'LICENSE.md'));

      // Copy binary if it exists
      if (fs.existsSync(artifactPath)) {
        fs.copyFileSync(artifactPath, path.join(platformDir, platform.binaryName));
        console.log(`✓ ${platform.name} (with binary)`);
      } else {
        console.log(`✓ ${platform.name} (no binary found at ${artifactPath})`);
      }

      successCount++;
    } catch (error) {
      console.error(`✗ ${platform.name}: ${error.message}`);
      errorCount++;
    }
  }

  console.log(`\nGenerated ${successCount} platform package(s)`);

  if (errorCount > 0) {
    console.error(`Failed to generate ${errorCount} package(s)`);
    process.exit(1);
  }

  console.log('Done!');
}

// Run
if (require.main === module) {
  main();
}

module.exports = { PLATFORMS, generatePackageJson, generateIndexJs, generateReadme };
