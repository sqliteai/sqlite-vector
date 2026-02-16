# Makefile for SQLite Vector Extension
# Supports compilation for Linux, macOS, Windows, Android and iOS

# customize sqlite3 executable with 
# make test SQLITE3=/opt/homebrew/Cellar/sqlite/3.49.1/bin/sqlite3
SQLITE3 ?= sqlite3

# Set default platform if not specified
ifeq ($(OS),Windows_NT)
	PLATFORM := windows
	HOST := windows
	CPUS := $(shell powershell -Command "[Environment]::ProcessorCount")
else
	HOST = $(shell uname -s | tr '[:upper:]' '[:lower:]')
	ifeq ($(HOST),darwin)
		PLATFORM := macos
		CPUS := $(shell sysctl -n hw.ncpu)
	else
		PLATFORM := $(HOST)
		CPUS := $(shell nproc)
	endif
endif

# Speed up builds by using all available CPU cores
MAKEFLAGS += -j$(CPUS)

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Wno-unused-parameter -I$(SRC_DIR) -I$(LIB_DIR)

# Directories
SRC_DIR = src
DIST_DIR = dist
LIB_DIR = libs
VPATH = $(SRC_DIR):$(LIB_DIR)
BUILD_DIR = build

# Files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst %.c, $(BUILD_DIR)/%.o, $(notdir $(SRC_FILES)))

# Platform-specific settings
ifeq ($(PLATFORM),windows)
	TARGET := $(DIST_DIR)/vector.dll
	LDFLAGS += -shared -static-libgcc
	# Create .def file for Windows
	DEF_FILE := $(BUILD_DIR)/vector.def
	STRIP = strip --strip-unneeded $@
else ifeq ($(PLATFORM),macos)
	TARGET := $(DIST_DIR)/vector.dylib
	ifndef ARCH
		LDFLAGS += -arch x86_64 -arch arm64
		CFLAGS += -arch x86_64 -arch arm64
	else
		LDFLAGS += -arch $(ARCH)
		CFLAGS += -arch $(ARCH)
	endif
	LDFLAGS += -dynamiclib -undefined dynamic_lookup -headerpad_max_install_names
	STRIP = strip -x -S $@
else ifeq ($(PLATFORM),android)
	ifndef ARCH # Set ARCH to find Android NDK's Clang compiler, the user should set the ARCH
		$(error "Android ARCH must be set to ARCH=x86_64, ARCH=arm64-v8a, or ARCH=armeabi-v7a")
	endif
	ifndef ANDROID_NDK # Set ANDROID_NDK path to find android build tools; e.g. on MacOS: export ANDROID_NDK=/Users/username/Library/Android/sdk/ndk/25.2.9519653
		$(error "Android NDK must be set")
	endif
	BIN = $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(HOST)-x86_64/bin
	ifneq (,$(filter $(ARCH),arm64 arm64-v8a))
		override ARCH := aarch64
		ANDROID_ABI := android26
	else ifeq ($(ARCH),armeabi-v7a)
		override ARCH := armv7a
		ANDROID_ABI := androideabi26
	else
		ANDROID_ABI := android26
	endif
	CC = $(BIN)/$(ARCH)-linux-$(ANDROID_ABI)-clang
	TARGET := $(DIST_DIR)/vector.so
	LDFLAGS += -lm -shared -Wl,-z,max-page-size=16384
	STRIP = $(BIN)/llvm-strip --strip-unneeded $@
else ifeq ($(PLATFORM),ios)
	TARGET := $(DIST_DIR)/vector.dylib
	SDK := -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=11.0
	LDFLAGS += -dynamiclib $(SDK) -headerpad_max_install_names
	CFLAGS += -arch arm64 $(SDK)
	STRIP = strip -x -S $@
else ifeq ($(PLATFORM),ios-sim)
	TARGET := $(DIST_DIR)/vector.dylib
	SDK := -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=11.0
	LDFLAGS += -arch x86_64 -arch arm64 -dynamiclib $(SDK) -headerpad_max_install_names
	CFLAGS += -arch x86_64 -arch arm64 $(SDK)
	STRIP = strip -x -S $@
else # linux
	TARGET := $(DIST_DIR)/vector.so
	LDFLAGS += -shared
	STRIP = strip --strip-unneeded $@
endif

# Windows .def file generation
$(DEF_FILE):
ifeq ($(PLATFORM),windows)
	@echo "LIBRARY vector.dll" > $@
	@echo "EXPORTS" >> $@
	@echo "    sqlite3_vector_init" >> $@
endif

# Make sure the build and dist directories exist
$(shell mkdir -p $(BUILD_DIR) $(DIST_DIR))

# Default target
extension: $(TARGET)
all: $(TARGET) 

# Loadable library
$(TARGET): $(OBJ_FILES) $(DEF_FILE)
	$(CC) $(OBJ_FILES) $(DEF_FILE) -o $@ $(LDFLAGS)
ifeq ($(PLATFORM),windows)
	# Generate import library for Windows
	dlltool -D $@ -d $(DEF_FILE) -l $(DIST_DIR)/vector.lib
endif
	# Strip debug symbols
	$(STRIP)

# Object files
$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -O3 -fPIC -c $< -o $@

test: $(TARGET)
	$(SQLITE3) ":memory:" -cmd ".bail on" ".load ./dist/vector" "SELECT vector_version();"

TEST_SRC = test/test_vector.c libs/sqlite3.c $(SRC_FILES)
unittest:
	$(CC) $(CFLAGS) -DSQLITE_CORE -O2 $(TEST_SRC) -o $(BUILD_DIR)/test_vector -lm -lpthread
	./$(BUILD_DIR)/test_vector

# Clean up generated files
clean:
	rm -rf $(BUILD_DIR)/* $(DIST_DIR)/* *.gcda *.gcno *.gcov *.sqlite

.NOTPARALLEL: %.dylib
%.dylib:
	rm -rf $(BUILD_DIR) && $(MAKE) PLATFORM=$*
	mv $(DIST_DIR)/vector.dylib $(DIST_DIR)/$@

define PLIST
<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\
<plist version=\"1.0\">\
<dict>\
<key>CFBundleDevelopmentRegion</key>\
<string>en</string>\
<key>CFBundleExecutable</key>\
<string>vector</string>\
<key>CFBundleIdentifier</key>\
<string>ai.sqlite.vector</string>\
<key>CFBundleInfoDictionaryVersion</key>\
<string>6.0</string>\
<key>CFBundlePackageType</key>\
<string>FMWK</string>\
<key>CFBundleSignature</key>\
<string>????</string>\
<key>CFBundleVersion</key>\
<string>$(shell make version)</string>\
<key>CFBundleShortVersionString</key>\
<string>$(shell make version)</string>\
<key>MinimumOSVersion</key>\
<string>11.0</string>\
</dict>\
</plist>
endef

define MODULEMAP
framework module vector {\
  umbrella header \"sqlite-vector.h\"\
  export *\
}
endef

LIB_NAMES = ios.dylib ios-sim.dylib macos.dylib
FMWK_NAMES = ios-arm64 ios-arm64_x86_64-simulator macos-arm64_x86_64
$(DIST_DIR)/%.xcframework: $(LIB_NAMES)
	@$(foreach i,1 2 3,\
		lib=$(word $(i),$(LIB_NAMES)); \
		fmwk=$(word $(i),$(FMWK_NAMES)); \
		mkdir -p $(DIST_DIR)/$$fmwk/vector.framework/Headers; \
		mkdir -p $(DIST_DIR)/$$fmwk/vector.framework/Modules; \
		cp src/sqlite-vector.h $(DIST_DIR)/$$fmwk/vector.framework/Headers; \
		printf "$(PLIST)" > $(DIST_DIR)/$$fmwk/vector.framework/Info.plist; \
		printf "$(MODULEMAP)" > $(DIST_DIR)/$$fmwk/vector.framework/Modules/module.modulemap; \
		mv $(DIST_DIR)/$$lib $(DIST_DIR)/$$fmwk/vector.framework/vector; \
		install_name_tool -id "@rpath/vector.framework/vector" $(DIST_DIR)/$$fmwk/vector.framework/vector; \
	)
	xcodebuild -create-xcframework $(foreach fmwk,$(FMWK_NAMES),-framework $(DIST_DIR)/$(fmwk)/vector.framework) -output $@
	rm -rf $(foreach fmwk,$(FMWK_NAMES),$(DIST_DIR)/$(fmwk))

xcframework: $(DIST_DIR)/vector.xcframework

AAR_ARM64 = packages/android/src/main/jniLibs/arm64-v8a/
AAR_ARM = packages/android/src/main/jniLibs/armeabi-v7a/
AAR_X86 = packages/android/src/main/jniLibs/x86_64/
aar:
	mkdir -p $(AAR_ARM64) $(AAR_ARM) $(AAR_X86)
	$(MAKE) clean && $(MAKE) PLATFORM=android ARCH=arm64-v8a
	mv $(DIST_DIR)/vector.so $(AAR_ARM64)
	$(MAKE) clean && $(MAKE) PLATFORM=android ARCH=armeabi-v7a
	mv $(DIST_DIR)/vector.so $(AAR_ARM)
	$(MAKE) clean && $(MAKE) PLATFORM=android ARCH=x86_64
	mv $(DIST_DIR)/vector.so $(AAR_X86)
	cd packages/android && ./gradlew clean assembleRelease
	cp packages/android/build/outputs/aar/android-release.aar $(DIST_DIR)/vector.aar

version:
	@echo $(shell sed -n 's/^#define SQLITE_VECTOR_VERSION[[:space:]]*"\([^"]*\)".*/\1/p' src/sqlite-vector.h)

# Help message
help:
	@echo "SQLite Vector Extension Makefile"
	@echo "Usage:"
	@echo "  make [PLATFORM=platform] [ARCH=arch] [ANDROID_NDK=\$$ANDROID_HOME/ndk/26.1.10909125] [target]"
	@echo ""
	@echo "Platforms:"
	@echo "  linux (default on Linux)"
	@echo "  macos (default on macOS)"
	@echo "  windows (default on Windows)"
	@echo "  android (needs ARCH to be set to x86_64, arm64-v8a, or armeabi-v7a and ANDROID_NDK to be set)"
	@echo "  ios (only on macOS)"
	@echo "  ios-sim (only on macOS)"
	@echo ""
	@echo "Targets:"
	@echo "  all			- Build the extension (default)"
	@echo "  clean			- Remove built files"
	@echo "  test			- Test the extension"
	@echo "  help			- Display this help message"
	@echo "  xcframework	- Build the Apple XCFramework"
	@echo "  aar			- Build the Android AAR package"

.PHONY: all clean test unittest extension help version xcframework aar
