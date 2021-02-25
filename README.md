# The SpV8 kernel

SpV8 is a SpMV kernel written in AVX-512. The research goal of SpV8 is to pursue optimal vectorization and regular computation pattern.

This is the artifact for our paper @ DAC '21: `SpV8: Pursuing Optimal Vectorization and Regular Computation Pattern in SpMV`

# Usage

## Dependencies & Recommend Environment:

1. Intel Processor with AVX-512 support
2. GCC 9
3. Intel Parallel Studio XE 2020.2
4. numactl (to set affinity)
5. powercpu (to set frequency)

## Compilation

```
make
```

## Run kernel

...

# Notes

This repo only contains two kernel for SpV8 and MKL. For other methods like CVR, ESB and CSR5, we reused kernels provided in [puckbee/pava](https://github.com/puckbee/pava). These kernel are collected from the original author. And we only modified their output code to simplify data collection.

We thank Biwei Xie, the author of CVR, for his informative discussion on running previous methods.

# License

The code is licensed with MIT Opensource License.

But note that, the dataset, Intel MKL and other previous research kernels are copyright by other entities.

