#!/bin/bash

# Exit immediately if a command fails
set -e

# Compile the program
/opt/homebrew/Cellar/llvm/19.1.6/bin/clang -O3 -fopenmp -o STREAM_CPU.out ./stream.c

echo "Compilation succeeded."

# Get the number of CPU cores (macOS specific)
cpuCores=$(sysctl -n hw.ncpu)

# Loop from 1 to the number of CPU cores
for ((i=1; i<=cpuCores; i++)); do
    echo "Running with $i thread(s)"
    
    # Set the number of threads for OpenMP
    export OMP_NUM_THREADS=$i
    
    # Run the program and save the output
    ./STREAM_CPU.out > "./STREAM-CPU-$i.txt"
done

echo "All runs completed successfully."
