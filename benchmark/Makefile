# SPDX-License-Identifier: GPL-2.0-only
# Build the BF16/FP16/TF32 GEMM benchmark. Requires CUDA toolkit + cuBLAS.
#
#   make            # build gemm
#   make run        # build and run with defaults (8192x8192, 20 iters)
#   make clean
#
# The single binary times all three input types (BF16, FP16, TF32), each
# with FP32 accumulation, and reports TFLOPS for each.

NVCC      ?= nvcc
ARCH      ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH)
LDLIBS    ?= -lcublas

TARGET := gemm

all: $(TARGET)

$(TARGET): gemm_fp16.cu
	$(NVCC) $(NVCCFLAGS) $< $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET) 8192 20

clean:
	rm -f $(TARGET)

.PHONY: all run clean