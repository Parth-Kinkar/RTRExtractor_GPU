# RTRExtractor GPU

A VRAM-native, GPU-accelerated implementation of the **RTRex** graph decomposition and clustering algorithm. This pipeline utilizes **CUDA** and **Thrust** to perform massively parallel triangle counting and density-based cluster extraction on large-scale graphs.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Linux (Ubuntu/Debian preferred) |
| **Compiler** | GCC/G++ with C++14 or higher support |
| **CUDA Toolkit** | CUDA 11.x or 12.x (Tested on CUDA 12.4) |
| **Build System** | CMake 3.18+ |

---

## 1. Compilation

This project uses CMake for build configuration. To compile the GPU executable, run the following commands from the root directory:

```bash
# Create and enter the build directory
mkdir -p tricore/build
cd tricore/build

# Generate the Makefile
# Target a specific GPU architecture if necessary (e.g., sm_86 for Ampere)
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..

# Compile using all available CPU cores
make -j$(nproc)
```

> **Note:** If your environment defaults to an older CUDA version, explicitly point CMake to the correct compiler:
> ```bash
> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.x/bin/nvcc
> ```

---

## 2. Dataset Preprocessing — Text to Binary

To eliminate PCIe transfer bottlenecks and maximize I/O speed, the algorithm requires graphs in a proprietary `.bin` format rather than raw text edge lists.

If you have a raw unweighted edge list (e.g., `graph.txt`), convert it first using the provided utility:

```bash
# General conversion syntax
./text2bin <input_text_file> <output_binary_file>

# Example: convert a toy graph
./text2bin ../datasets/test_graph.txt ../datasets/test_graph.bin
```

---

## 3. Execution

Once the graph is in `.bin` format, pass it directly to the main GPU extraction algorithm:

```bash
# General execution syntax
./tric_gpu <path_to_binary_graph> <mode>

# Example: run with debug logging enabled
./tric_gpu ../datasets/test_graph.bin debug
```

---

## Expected Output

The execution outputs an **architectural bottleneck profile** with the following metrics:

| Metric | Description |
|---|---|
| **VRAM Malloc/Free Time** | Time spent allocating and freeing GPU memory |
| **PCIe Transfer Time** | Host ↔ Device data transfer overhead |
| **GPU Compute Time** | Core parallel computation time on device |
| **CPU Compute Time** | Serial/host-side processing time |
| **Total Non-trivial Clustered Vertices** | Count of vertices in meaningful dense clusters |
