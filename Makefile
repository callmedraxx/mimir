CC = gcc
CFLAGS = -O3 -march=native -mavx2 -mfma -Wall -Wextra -std=c11 -fopenmp -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm -fopenmp -lpthread
DEBUG_FLAGS = -g -O0 -Wall -Wextra -std=c11 -DDEBUG -fsanitize=address

SRC_DIR = src
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

CHECKPOINT = data/mimir.brain

# Main targets
.PHONY: all clean clean-all test debug force-train

all: $(BUILD_DIR)/mimir

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/mimir: $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

debug: CFLAGS = $(DEBUG_FLAGS)
debug: clean $(BUILD_DIR)/mimir

test: $(BUILD_DIR)/mimir
	./$(BUILD_DIR)/mimir --test

# Browser build for Workbench Lab's Mimir bench: the same C, compiled to
# WebAssembly, with every function traced on entry and exit.
WASI_SDK ?= /opt/wasi-sdk-25.0-x86_64-linux
WASM_FLAGS = --target=wasm32-wasip1 --sysroot=$(WASI_SDK)/share/wasi-sysroot \
             -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -DMIMIR_WASM \
             -Wno-declaration-after-statement
wasm: $(BUILD_DIR)/mimir.wasm
$(BUILD_DIR)/mimir.wasm: $(SRCS) src/mimir.h tools/wasm_instrument.py
	@mkdir -p $(BUILD_DIR)
	python3 tools/wasm_instrument.py src $(BUILD_DIR)/wasm-src
	$(WASI_SDK)/bin/clang $(WASM_FLAGS) $(BUILD_DIR)/wasm-src/*.c -o $@ -lm \
		-Wl,--export-table -Wl,-z,stack-size=1048576

# Remove compiled artifacts only — trained knowledge in data/ is preserved.
clean:
	rm -rf $(BUILD_DIR)

# Full reset: remove compiled artifacts AND all saved brains (forces retraining).
# WARNING: this permanently deletes all trained knowledge — gate brain,
# alphabet brain, and vocabulary. You will have to reteach everything.
clean-all:
	rm -rf $(BUILD_DIR) $(CHECKPOINT) data/mimir_abc.brain data/mimir_abc.vocab \
	       data/mimir_hippo.dat data/mimir_training.log

# Build and run, forcing a full retrain regardless of any saved checkpoint.
force-train: $(BUILD_DIR)/mimir
	./$(BUILD_DIR)/mimir --force-train
