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
    LDFLAGS += -shared
    # Create .def file for Windows
    DEF_FILE := $(BUILD_DIR)/vector.def
else ifeq ($(PLATFORM),macos)
    TARGET := $(DIST_DIR)/vector.dylib
    LDFLAGS += -arch x86_64 -arch arm64 -dynamiclib -undefined dynamic_lookup
    CFLAGS += -arch x86_64 -arch arm64
else ifeq ($(PLATFORM),android)
    # Set ARCH to find Android NDK's Clang compiler, the user should set the ARCH
    ifeq ($(filter %,$(ARCH)),)
        $(error "Android ARCH must be set to ARCH=x86_64 or ARCH=arm64-v8a")
    endif
    # Set ANDROID_NDK path to find android build tools
    # e.g. on MacOS: export ANDROID_NDK=/Users/username/Library/Android/sdk/ndk/25.2.9519653
    ifeq ($(filter %,$(ANDROID_NDK)),)
        $(error "Android NDK must be set")
    endif

    BIN = $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(HOST)-x86_64/bin
    PATH := $(BIN):$(PATH)

    ifneq (,$(filter $(ARCH),arm64 arm64-v8a))
        override ARCH := aarch64
    endif

    CC = $(BIN)/$(ARCH)-linux-android26-clang
    TARGET := $(DIST_DIR)/vector.so
    LDFLAGS += -shared
else ifeq ($(PLATFORM),ios)
    TARGET := $(DIST_DIR)/vector.dylib
    SDK := -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=11.0
    LDFLAGS += -dynamiclib $(SDK)
    CFLAGS += -arch arm64 $(SDK)
else ifeq ($(PLATFORM),isim)
    TARGET := $(DIST_DIR)/vector.dylib
    SDK := -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=11.0
    LDFLAGS += -arch x86_64 -arch arm64 -dynamiclib $(SDK)
    CFLAGS += -arch x86_64 -arch arm64 $(SDK)
else # linux
    TARGET := $(DIST_DIR)/vector.so
    LDFLAGS += -shared
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

# Object files
$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -O3 -fPIC -c $< -o $@

test: $(TARGET)
	$(SQLITE3) ":memory:" -cmd ".bail on" ".load ./$<" "SELECT vector_version();"

# Clean up generated files
clean:
	rm -rf $(BUILD_DIR)/* $(DIST_DIR)/* *.gcda *.gcno *.gcov *.sqlite

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
	@echo "  android (needs ARCH to be set to x86_64 or arm64-v8a and ANDROID_NDK to be set)"
	@echo "  ios (only on macOS)"
	@echo "  isim (only on macOS)"
	@echo ""
	@echo "Targets:"
	@echo "  all       				- Build the extension (default)"
	@echo "  clean     				- Remove built files"
	@echo "  test					- Test the extension"
	@echo "  help      				- Display this help message"

.PHONY: all clean test extension help
