# The SpV8 kernel

SpV8 is a SpMV kernel written in AVX-512. The research goal of SpV8 is to pursue optimal vectorization and regular computation pattern.

This is the artifact for our paper @ DAC '21: `SpV8: Pursuing Optimal Vectorization and Regular Computation Pattern in SpMV`

# Usage

## Dependencies & Recommend Environment

1. Intel Xeon Processor with AVX-512 support
2. GCC 9
3. Intel Parallel Studio XE 2020.2
4. numactl (to set numa affinity)
5. cpupower (to set frequency, other available tools should also work)

## Compilation

```
make
```

## Run kernel

Before running, CPU frequency should be fixed to eliminate CPU Turbo effects.

The binaries have the same parameter list as follows:

```bash
bin/spmv_(kernel) [loop_count] [use_optimize?] [thread_count]
```

1. `loop_count`: Number of iterations of SpMV
2. `use_optimize?`: Taking effects only on MKL binary. It is used to activate MKL's Inspector-Executor Optimization.
3. `thread_count`: Number of OpenMP threads

Example for running SpV8:

```bash
numactl --cpunodebind=1 --membind=1 bin/spmv_spv8 1000 1 8
```

## How to feed matrix to kernel

In our experiments, we used a custom but straightforward data format to store CSR matrix. And once we execute the binary, it will search for the following data files **under the work directory**:

```
info.txt : Contains number of NNZ, rows, cols
rowb.txt : 0-based NNZ index for each row begin
rowe.txt : 0-based NNZ index for each row end
nnz.txt  : NNZ list
col.txt  : 0-based column index for each NNZ
x.txt    : A random vector x, used for SpMV
ans.txt  : Used to check answer
```

Meanwhile, **we also provide two script** in `contrib` for you to generate data files from Matlab Matrix Format or Matrix Market Format.

## Dataset

All the matrices we used in our benchmark are listed in `contrib/data.txt`, and their files are publicly available on [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

# Notes

This repo only contains two kernel for SpV8 and MKL. For other methods like CVR, ESB and CSR5, we reused kernels provided in [puckbee/CVR](https://github.com/puckbee/CVR) and [puckbee/pava](https://github.com/puckbee/pava). These kernels are collected from the original authors. And we only modified their output code to simplify data collection.

We thank Biwei Xie, the author of CVR, for his kind and informative discussion on running previous methods.

# License

The code is licensed with MIT Opensource License.

But note that, the dataset, Intel MKL and other previous research kernels are copyright by other entities.
