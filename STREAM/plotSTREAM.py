import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

formats = [None, ".", "*", "x", "-", "/", "o", "+", "&", "v", "^", "<", ">"]


plt.rcParams.update({
    "text.usetex": True,                  
    "font.family": "serif",                
    "font.serif": ["Computer Modern"],    
    "font.size": 10,                      
    "axes.labelsize": 12,                 
    "axes.titlesize": 12,                 
    "legend.fontsize": 10,              
    "xtick.labelsize": 10,                
    "ytick.labelsize": 10,                
})

# Function to parse files and extract data
def parse_stream_results(directory):
    groups = ["M1", "M2", "M3", "M4"]
    metrics = ["Copy", "Scale", "Add", "Triad"]
    cpu_bandwidth = []
    gpu_bandwidth = []

    for group in groups:
        cpu_file = os.path.join(directory, f"{group}-CPU.txt")
        gpu_file = os.path.join(directory, f"{group}-GPU.txt")

        # Parse CPU file
        with open(cpu_file, "r") as f:
            content = f.read()
            cpu_data = [
                float(match.group(2)) / 1000
                for match in re.finditer(r"^\s*(Copy|Scale|Add|Triad):\s*([\d.]+)", content, re.M)
            ]
            cpu_bandwidth.append(cpu_data)

        # Parse GPU file
        toSI = (1 << 30) / 1e9
        with open(gpu_file, "r") as f:
            content = f.read()
            gpu_data = [
                float(match.group(2)) * toSI
                for match in re.finditer(r"^\s*(Copy|Scale|Add|Triad):\s*([\d.]+)", content, re.M)
            ]
            gpu_bandwidth.append(gpu_data)

    return metrics, groups, np.array(cpu_bandwidth).T, np.array(gpu_bandwidth).T


# Visualization function
def plot_bandwidth(title, groups, metrics, cpu_bandwidth, gpu_bandwidth, ylabel, advertised_bandwidth):
    x = np.arange(len(groups))
    bar_width = 0.1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each metric
    for i, metric in enumerate(metrics):  # CPU
        ax.bar(
            x + i * bar_width,
            cpu_bandwidth[i],
            bar_width,
            label=metric + " (CPU)",
            hatch=formats[i],
        )

    for i, metric in enumerate(metrics):  # GPU
        ax.bar(
            x + (i + len(metrics)) * bar_width,
            gpu_bandwidth[i],
            bar_width,
            label=metric + " (GPU)",
            hatch=formats[i + len(metrics)],
        )

    # Plot advertised bandwidth as a line
    for idx, adv_bw in enumerate(advertised_bandwidth):
        cluster_center = x[idx] + (len(metrics) * bar_width - 0.5 * bar_width)
        ax.hlines(
            adv_bw,
            cluster_center - 4 * bar_width,
            cluster_center + 4 * bar_width,
            colors="red",
            linestyles="--",
            label="Theoretical Bandwidth" if idx == 0 else None,
        )

    # Customize the plot
    ax.set_xlabel("Chip Models", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + 3.5 * bar_width)
    ax.set_xticklabels(groups, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()
    ax.figure.savefig("STREAM.pdf", format="pdf", dpi=300, bbox_inches="tight")


# Main script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    advertised_bandwidth = [67, 100, 100, 120]
    metrics, groups, cpu_bandwidth, gpu_bandwidth = parse_stream_results(directory)

    plot_bandwidth(
        title="",
        groups=groups,
        metrics=metrics,
        cpu_bandwidth=cpu_bandwidth,
        gpu_bandwidth=gpu_bandwidth,
        ylabel="Bandwidth (GB/s)",
        advertised_bandwidth=advertised_bandwidth,
    )