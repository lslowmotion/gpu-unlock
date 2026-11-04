# SPDX-License-Identifier: GPL-2.0-only
# Build the BF16 GEMM benchmark. Requires CUDA toolkit + cuBLAS.
#
#   make            # build gemm_bf16
#   make run        # build and run with defaults (8192x8192, 20 iters)
#   make clean

NVCC      ?= nvcc
ARCH      ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH)
LDLIBS    ?= -lcublas

TARGET := gemm_bf16

all: $(TARGET)

$(TARGET): gemm_fp16.cu
	$(NVCC) $(NVCCFLAGS) $< $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET) 8192 20

clean:
	rm -f $(TARGET)

.PHONY: all run clean