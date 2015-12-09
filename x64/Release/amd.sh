#!/bin/bash

export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export GPU_USE_SYNC_OBJECTS=1
./dagSimCL-linux64

# gdb dagSimCL-linux64-debug
