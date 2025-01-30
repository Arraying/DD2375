# Apple Silicon Matrix Multiplication Benchmarks

Repository for software for a research project aiming to answer the research question:
> What is the computational performance (in FLOP/s) and power usage (in W) of single-precision matrix-matrix multiplication on an Apple M1 - M4 SoC using pure CPU, CPU accelerators, and GPU?

Kind thanks to the open-source repositories supporting this research:
* Some Metal shaders from: https://github.com/bkvogel/metal_performance_testing
* The OpenMP block multiplication implementation from: https://github.com/dmitrydonchenko/Block-Matrix-Multiplication-OpenMP

## Environment

This project requires Xcode, C++20 and `libomp`. The following software and versions have been found to work:
| Name | M1 | M2 | M3 | M4 |
| --- | --- | --- | --- | --- |
| macOS | 14.7.2    | 15.1.1    | 15.2      | 15.1.1 |
| CMake | 3.28      | 3.31.4    | 3.31.4    | 3.31.4 |
| Make  | 3.81      | 3.81      | 3.81      | 3.81   |
| Xcode | 14.0.1, 15.4    | 16.2      | 16.2      | 16.2  |
| AppleClang | 14.0.0    | 16.0.0    | 16.0.0 | 16.0.0 |
| Clang     |    19.1.7       |    19.1.7      |    19.1.6       |    19.1.7       |
| Metal | 31001.643 | 32023.404 | 32023.404 | 32023.404 |
| libomp | 19.1.4   | 19.1.7    | 19.1.6    | 19.1.7    |


## Benchmarking

The Benchmarking python scripts handle the building, execution and results according to the list in `config.py`.

``` shell

mkdir data
mkdir out
mkdir results

cd control 
python3 -m venv .venv
source ./.venv/bin/activate
# In virtual python evnironment now

pip install -r ./requirements.txt

python ./make_matrices.py &&  ls ../data

sudo python ./benchmark.py  # may take longer time
ls ../out

python ./make_analysis.py
ls ../results

```

The benchmarking can be performed automatically using the `./control` suite.
This test suite uses Python, a virtual environment with Python 3.9.6 and the packages in `./control/requirements.txt` works.

If CMAKE can not find METALC, please try:

``` shell
sudo bash -c "export METALC="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal" && python ./benchmark.py"
```

### Running the Suite

Ensure to create the `./out` and `./results` directories.
Enter the virtual environment, cwd into `./control` and run `sudo python3 benchmark.py`. 
Superuser privileges are **required** as `powermetrics` requires them.

### Analysing the Results.

Enter the virtual environment, cwd into `./control` and run `python3 make_analysis.py`.
This will create graphs inside the `./results` directory.
LaTeX tabular data will also be printed to stdout.


## Manual Compilation

Please generate the matrices data according to **Benchmarking** section above.

1. `mkdir build && cd build/`
2. `cmake -DMATRIX_N=<N> -DCMAKE_BUILD_TYPE=Release ..`, replace `<N>` with a number such that the files `./data/matrix-<N>-0.float32` and `./data/matrix-<N>-1.float32` exist
3. `make`

## Manual Execution

For CPU-based builds, cwd into `build` and execute the binary.
For GPU-based builds, cwd into `build/XXX.app/Contents/MacOS` (where `XXX` is the name of an executable) and execute the binary.


## STREAM

To test the bandwidth of unified memory architecture of M-Series chips, STREAM test is used. And a Metal GPU version was ported from a [CUDA/HIP version](https://github.com/KTH-ScaLab/multi-gpu-comm/blob/master/tools/stream/stream_cpugpu.cpp). 

### Usage

Change the clang path in [STREAM-CPU.sh](./STREAM/STREAM-CPU.sh). And execute the scripts.

Follow the commands in [metalBuild.txt](./STREAM/metalBuild.txt). And execute the `streamMetal.out`

