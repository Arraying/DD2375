import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools

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

prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
shifted_colors = list(itertools.islice(itertools.cycle(prop_cycle), 2, len(prop_cycle) + 2))


SoC = ["M1", "M2", "M3", "M4"]

sizes = [2048, 4096, 8192, 16384]

RESULTS_FOLDER = "results"
RAW_DATA_FOLDER = "out"

implementations = ["baseline", "omp"]

labels = ["CPU-Single", "CPU-OMP"] 

formats = [None, ".", "*", "x", "-", "/", "o", "+", "&", "v", "^", "<", ">"]



def setup(path):
    os.system(f"mkdir -p {path}")


def get_powers(matrix_size, trial, implementation):
    cpu, pwr, gpu = None, 0, None

    extract = lambda row: int(row.split(" ")[2])
    with open(RAW_DATA_FOLDER + f"/{matrix_size}x{matrix_size}/{trial}/{implementation}/power.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("CPU Power"):
                cpu = int(extract(line))
            # elif line.startswith("P-Cluster Power"):
            #     pwr = int(extract(line))
            elif line.startswith("GPU Power"):
                gpu = int(extract(line))
    return cpu, pwr, gpu

def getPower(path, matrix_size, trial, implementation):
    cpu, pwr, gpu = None, 0, None

    extract = lambda row: int(row.split(" ")[2])
    with open(path + "/" + RAW_DATA_FOLDER + f"/{matrix_size}x{matrix_size}/{trial}/{implementation}/power.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("CPU Power"):
                cpu = int(extract(line))
            # elif line.startswith("P-Cluster Power"):
            #     pwr = int(extract(line))
            elif line.startswith("GPU Power"):
                gpu = int(extract(line))
    return cpu, pwr, gpu


def get_powers_data(implementation):
    raw = [[get_powers(size, trial, implementation) for trial in range(1, 6)] for size in sizes]
    processed = np.array(raw)
    return processed


def get_timing(matrix_size, trial, implementation):
    with open(RAW_DATA_FOLDER + f"/{matrix_size}x{matrix_size}/{trial}/{implementation}/timing.txt", "r") as f:
        return int(f.read()) / 1_000_000  # time in ms


def get_timing_data(implementation):
    raw = [[get_timing(size, trial, implementation) for trial in range(1, 6)] for size in sizes]
    processed = np.array(raw)
    return np.mean(processed, axis=1), np.std(processed, axis=1)


def plot_timing_implementation(implementation, label, fmt, color):
    y, err = get_timing_data(implementation)
    plt.errorbar(sizes, y, yerr=err, fmt=fmt, color=color, label=label)
    print(label, "&", " & ".join([f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(y, err)]), "\\\\")


def plot_timing():
    plt.cla()
    plt.figure(figsize=(7, 9))
    print("TIMING [ms]")

    plot_timing_implementation("baseline", "Naive", "o", "C0")
    plot_timing_implementation("omp", "Block Multiplication", "s", "C3")
    plot_timing_implementation("blas", "BLAS", "v", "C1")
    plot_timing_implementation("dsp", "vDSP", "^", "C2")
    plot_timing_implementation("gpu_baseline", "Naive Shader", "D", "C4")
    plot_timing_implementation("gpu_nv", "Cutlass-Style Shader", "h", "C5")
    plot_timing_implementation("gpu_mps", "MPS", "*", "C6")
    plt.xlabel("Matrix size")
    plt.xticks(sizes)
    plt.ylabel("Average multiplication time [ms], log scale, lower is better")
    plt.yscale("log")
    plt.legend(title="Implementation")
    plt.savefig(RESULTS_FOLDER + "/timing.png", dpi=300)


def plot_power_implementation(implementation, label, fmt, color):
    all_data = np.sum(get_powers_data(implementation), axis=2)
    y = np.mean(all_data, axis=1)
    err = np.std(all_data, axis=1)
    plt.errorbar(np.arange(len(sizes)), y, yerr=err, fmt=fmt, color=color, label=label)
    # print(label, "&", " & ".join([f"{m:.2f} & {s:.2f}" for m, s in zip(y, err)]), "\\\\")


def plot_power():
    plt.cla()
    print("\n\nPOWER [mW]")

    # plot_power_implementation("baseline", "Naive", "o", "C0")
    # plot_power_implementation("omp", "Block Multiplication", "s", "C3")
    plot_power_implementation("blas", "BLAS", "v", "C1")
    plot_power_implementation("dsp", "vDSP", "^", "C2")
    plot_power_implementation("gpu_baseline", "Naive Shader", "D", "C4")
    plot_power_implementation("gpu_nv", "Cutlass-Style Shader", "h", "C5")
    plot_power_implementation("gpu_mps", "MPS", "*", "C6")
    plt.xlabel("Matrix size")
    plt.xticks(np.arange(len(sizes)))
    plt.ylabel("Average power dissipation [mW], lower is better")
    plt.legend(title="Implementation")
    # plt.savefig(RESULTS_FOLDER + "/power.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

def plot_power_allchips(chipPath, sizes):
    implementations = ["baseline", "omp", "blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    labels = ["CPU-Single", "CPU-OMP", "CPU-Accelerate", "GPU-Naive", "GPU-CUTLASS", "GPU-MPS"] 

    fmt = ["o", "s", "v", "^", "D", "h", "*"]

    print("\n\nPOWER-ALL")


    fig, axs = plt.subplots(1, len(SoC), figsize=(10, 5), sharey=True)

    for i, chip in enumerate(SoC):
        for j, implementation in enumerate(implementations):
            if implementation in ["baseline", "omp"]:
                raw = [[getPower(chipPath[i], size, trial, implementation) for trial in range(1, 6)] for size in sizes[0:-2]]
            else:
                raw = [[getPower(chipPath[i], size, trial, implementation) for trial in range(1, 6)] for size in sizes]
            processed = np.array(raw)
            all_data = np.sum(processed, axis=2)
            y = np.mean(all_data, axis=1)
            err = np.std(all_data, axis=1)
            axs[i].errorbar(np.arange(len(raw)), y, yerr=err, fmt=fmt[j], label=labels[j])

        axs[i].set_title(f"{chip}", fontsize=12, loc="left")
        axs[i].set_xticks(np.arange(len(sizes)), sizes)
        
        axs[i].set_yticks(np.arange(0, 24000, 2000))
        # axs[i].set_yticks([])
        axs[i].grid(True, linestyle="--", alpha=0.5)


        if i == 0:
            axs[i].set_ylabel("Average Power Dissipation [W]")

            axs[i].set_yticklabels(np.arange(0, 24, 2))
            # axs[i].legend(title="Implementation", loc="upper left")
    
    fig.legend(
        labels=labels,
        # title="Implementation",
        loc="lower center",
        ncol=len(implementations),
        # bbox_to_anchor=(0.5, -0.2),
        frameon=False
    )

    plt.tight_layout(rect=[0.00, 0.09, 0.95, 1])
    fig.text(0.5, 0.07, "Matrix Size", ha='center', fontsize=14)
    plt.savefig(f"{RESULTS_FOLDER}/Power.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

def plot_power_allchips_bar(chipPath, sizes):
    implementations = ["baseline", "omp", "blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    labels = ["CPU-Single", "CPU-OMP", "CPU-Accelerate", "GPU-Naive", "GPU-CUTLASS", "GPU-MPS"]
    barWidth = 0.15  # 控制每个 implementation 的柱子宽度
    
    print("\n\nPOWER-ALL")

    fig, axs = plt.subplots(1, len(SoC), figsize=(12, 5), sharey=True)
    x = np.arange(len(sizes))  # x 轴位置，对应 matrix sizes

    for i, chip in enumerate(SoC):
        ax = axs[i]
        ax.set_title(f"{chip}", fontsize=12, loc="left")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)

        if i == 0:
            ax.set_ylabel("Power Dissipation [mW]")
        
        for j, implementation in enumerate(implementations):
            if implementation in ["baseline", "omp"]:
                raw = [[getPower(chipPath[i], size, trial, implementation) for trial in range(1, 6)] for size in sizes[0:-2]]
            else:
                raw = [[getPower(chipPath[i], size, trial, implementation) for trial in range(1, 6)] for size in sizes]
            
            processed = np.array(raw)
            all_data = np.sum(processed, axis=2)
            y = np.mean(all_data, axis=1)
            err = np.std(all_data, axis=1)
            
            x_offset = [idx + (j + 0.5 - len(implementations) / 2) * barWidth for idx in range(len(y))]
            ax.bar(x_offset, y, width=barWidth, yerr=err, capsize=3, label=labels[j], hatch=formats[j])
    
    fig.legend(labels=labels, loc="lower center", ncol=len(implementations), frameon=False)
    plt.subplots_adjust(left=0.12, bottom=0.12, top=0.88)
    plt.tight_layout(rect=[0.00, 0.08, 0.95, 1])
    # fig.text(0.02, 0.5, "Power Dissipation [mW]", va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.07, "Matrix Size", ha='center', fontsize=14)
    plt.savefig(f"{RESULTS_FOLDER}/Power_bar.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()




def getFlops(path, matrix_size, trial, implementation):
    getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000  # time in ms
    flop = num_flop(matrix_size) / 1_000_000_000
    time = getTime(path, matrix_size, trial, implementation) / 1_000  # conversion: seconds
    flops = flop / time
    return flops

def plot_efficiency_allchips(chipPath, sizes):
    # Plot GFLOPS/W for all chips
    implementations = ["baseline", "omp", "blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    labels = ["CPU-Single", "CPU-OMP", "CPU-Accelerate", "GPU-Naive", "GPU-CUTLASS", "GPU-MPS"]

    fmt = ["o", "s", "v", "^", "D", "h", "*"]

    print("\n\nGFLOPS/W-ALL")

    fig, axs = plt.subplots(1, len(SoC), figsize=(10, 5), sharey=True)

    for i, chip in enumerate(SoC):


        for j, implementation in enumerate(implementations):

            if implementation in ["baseline", "omp"]:
                sizeCount = len(sizes) - 2
            else:
                sizeCount = len(sizes)

            resultImpl = []

            for k, matrix_size in enumerate(sizes):
                if k == sizeCount:
                    break

                power = [getPower(chipPath[i], matrix_size, trial, implementation) for trial in range(1, 6)]
                watts = np.sum(np.array(power), axis=1) / 1_000  # conversion: milli watt to watt, total cpu+gpu
                # averaged_watts = np.mean(watts, axis=0) # 1d array

                flop = num_flop(matrix_size) / 1_000_000_000 #gflop

                # get time 
                getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000.0  # time in ms
                times = [getTime(chipPath[i], matrix_size, trial, implementation) / 1000.0 for trial in range(1, 6)]
                flops = np.full(len(times), flop) / times
       
                flops_per_watt = flops / watts

                mean = np.mean(flops_per_watt, axis=0)
                std = np.std(flops_per_watt, axis=0)

                resultImpl.append((mean, std))

            axs[i].errorbar(np.arange(sizeCount), [resultImpl[k][0] for k in range(sizeCount)], yerr=[resultImpl[k][1] for k in range(sizeCount)], fmt=fmt[j], label=labels[j])
              

       
        
        axs[i].set_title(f"{chip}", fontsize=12, loc="left")
        axs[i].set_xticks(np.arange(len(sizes)))
        axs[i].set_xticklabels(sizes)
        axs[i].set_yscale("log")
        
        axs[i].grid(True, linestyle="--", alpha=0.5)

        if i == 0:
            axs[i].set_ylabel("Efficiency (GFLOPS/W)")

    fig.legend(
        labels=labels,
        loc="lower center",
        ncol=len(implementations),
        frameon=False
    )

    plt.tight_layout(rect=[0.00, 0.09, 0.95, 1])
    fig.text(0.5, 0.07, "Matrix Size", ha='center', fontsize=14)
    plt.savefig(f"{RESULTS_FOLDER}/Efficiency.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()




def plot_granular_values(matrix_size):
    raw = [[get_powers(matrix_size, i, implementation) for i in range(1, 6)] for implementation in implementations]
    processed = np.array(raw)
    return processed[:, :, 0], processed[:, :, 1], processed[:, :, 2]


def plot_granular(matrix_size):
    plt.cla()
    print("\n\nGRANULAR [mW]", matrix_size)
    cpu, _, gpu = plot_granular_values(matrix_size)
    cpu_y, gpu_y = np.mean(cpu, axis=1), np.mean(gpu, axis=1)
    cpu_err, gpu_err = np.std(cpu, axis=1), np.std(gpu, axis=1)
    cpu_err_lower = np.clip(cpu_err, None, cpu_y)
    gpu_err_lower = np.clip(gpu_err, None, gpu_y)
    for i, implementation in enumerate(implementations):
        print(labels[i], "&",
              "%.2f" % cpu_y[i], "&", "%.2f" % cpu_err[i], "&",
              "%.2f" % gpu_y[i], "&", "%.2f" % gpu_err[i], "\\\\")

    n_categories = len(implementations)
    group_height = 0.6
    bar_height = group_height / 3
    category_spacing = 0.4
    y_positions = np.arange(n_categories) * (group_height + category_spacing)
    y_positions = y_positions[::-1]
    y_positions_eff = y_positions + bar_height / 2
    y_positions_pwr = y_positions
    y_positions_gpu = y_positions - bar_height / 2

    plt.figure(figsize=(13, 6))
    plt.barh(y_positions_eff, cpu_y, xerr=[cpu_err_lower, cpu_err], height=bar_height,
             label="CPU", hatch=None)
    
    plt.barh(y_positions_gpu, gpu_y, xerr=[gpu_err_lower, gpu_err], height=bar_height,
             label="GPU", hatch="|")
    plt.xlabel("Average power dissipation per component [mW], lower is better")
    plt.ylabel("Implementation")
    plt.yticks(y_positions, labels)
    plt.legend(title="Components", loc="lower right")
    plt.savefig(RESULTS_FOLDER + f"/granular-{matrix_size}x{matrix_size}.png", dpi=300)


def num_flop(matrix_size):
    return matrix_size * matrix_size * ((2 * matrix_size) - 1)


def plot_flop_values():
    raw = [[[get_powers(matrix_size, i, implementation) for i in range(1, 6)]
            for matrix_size in sizes] for implementation in implementations]
    watts = np.sum(np.array(raw), axis=3) / 1_000  # conversion: milli watt to watt
    flops = np.array([num_flop(matrix_size) for matrix_size in sizes])[:, np.newaxis] / 1_000_000_000
    flops_per_watt = flops / watts
    return np.mean(flops_per_watt, axis=2), np.std(flops_per_watt, axis=2)


def plot_flop():
    plt.cla()
    print("\n\nGFLOP per W")
    components = [f"{n}x{n}" for n in sizes]
    y, y_err = plot_flop_values()

    num_categories = len(labels)
    num_components = len(components)
    bar_width = 0.12
    group_spacing = 0.6
    component_spacing = bar_width
    x_base = np.arange(num_categories) * (num_components * bar_width + group_spacing)
    x_positions = [x_base + i * component_spacing for i in range(num_components)]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, component in enumerate(components):
        ax.bar(x_positions[i], y[:, i], yerr=y_err[:, i], width=bar_width, label=component, hatch=formats[i])
    for i in range(len(implementations)):
        print(labels[i], "&", " & ".join([f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(y[i, :], y_err[i, :])]), "\\\\")

    ax.set_xlabel("Implementations")
    ax.set_ylabel("GFLOP per W")
    ax.set_xticks(x_base + (num_components - 1) * bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend(title="Matrix Size", loc="upper left")
    plt.savefig(RESULTS_FOLDER + "/efficiency.png", dpi=300)


def calculate_flops():
    plt.cla()
    print("\n\nGFLOPS")
    components = [f"{n}x{n}" for n in sizes]

    num_categories = len(labels)
    num_components = len(components)
    bar_width = 0.12
    group_spacing = 0.6
    component_spacing = bar_width
    x_base = np.arange(num_categories) * (num_components * bar_width + group_spacing)
    x_positions = [x_base + i * component_spacing for i in range(num_components)]
    fig, ax = plt.subplots(figsize=(12, 6))

    y = None
    y_err = None

    for i, implementation in enumerate(implementations):
        raw = [[get_timing(size, trial, implementation) for trial in range(1, 6)] for size in sizes]
        times = np.array(raw) / 1_000  # conversion: seconds
        flop = np.array([num_flop(matrix_size) for matrix_size in sizes]) / 1_000_000_000  # conversion: giga
        flops = flop[:, np.newaxis] / times
        mean = np.mean(flops, axis=1) 
        std = np.std(flops, axis=1)
        y = np.vstack([y, mean]) if y is not None else mean
        y_err = np.vstack([y_err, std]) if y_err is not None else std

        print(labels[i], "&", " & ".join([f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(mean, std)]), "\\\\")

    for i, component in enumerate(components):
        ax.bar(x_positions[i], y[:, i], yerr=y_err[:, i], width=bar_width, label=component, hatch=formats[i])

    ax.set_xlabel("Implementations")
    ax.set_ylabel("GFLOPS")
    ax.set_xticks(x_base + (num_components - 1) * bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend(title="Matrix Size", loc="upper left")
    plt.savefig(RESULTS_FOLDER + "/GFLOPS.png", dpi=300)


def calculate_flops_allchips_cpu(chipPath, size):

    implementations = ["baseline", "omp"]
    labels = ["CPU-Single", "CPU-OMP"] 

    print("\n\nGFLOPS-all")
    component = f"{size}x{size}" 

    flop = num_flop(size) / 1_000_000_000  # conversion: giga

    getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000  # time in ms

    results = []
    for i, chip in enumerate(SoC):
        chip_results = []
        for j, implementation in enumerate(implementations):
            raw = [getTime(chipPath[i], size, trial, implementation) for trial in range(1, 6)] 
            times = np.array(raw) / 1_000  # conversion: seconds
            flops = np.full(5, flop) / times

            mean = np.mean(flops, axis=0) 
            std = np.std(flops, axis=0)
            chip_results.append((mean, std))

        results.append(chip_results)

    bar_width = 0.3
    fig, axs = plt.subplots(len(SoC), 1, figsize=(6, 6), sharex=True)
    handles = []

    for i, (chip, ax) in enumerate(zip(SoC, axs)):
        means = [results[i][j][0] for j in range(len(implementations))]
        stds = [results[i][j][1] for j in range(len(implementations))]

        # Plot horizontal bar chart with hatches
        y_positions = np.arange(len(implementations))
        for j, (mean, std) in enumerate(zip(means, stds)):
            bar = ax.barh(
                y_positions[len(y_positions)-j - 1], mean, xerr=std, height=bar_width, alpha=0.8, 
                label=labels[j] if i == 0 else None, hatch=formats[j]
            )
            if i == 0:  # Collect legend handles only for the first subplot
                handles.append(bar)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        ax.set_title(f"{chip}", fontsize=10, loc="left")

        if i == len(SoC) - 1:
            ax.set_xlabel(f"GFLOPS-{size}x{size}", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        handles=[h[0] for h in handles],
        labels=labels,
        loc="upper right", ncol=1,  frameon=True)
    

    # Global adjustments
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/GFLOPS-{component}-cpu.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

def calculate_flops_allchips_acc(chipPath, size):

    implementations = ["blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    labels = ["CPU-Accelerate", "GPU-Naive", "GPU-CUTLASS", "GPU-MPS"] 

    print("\n\nGFLOPS-all")
    component = f"{size}x{size}" 

    flop = num_flop(size) / 1_000_000_000  # conversion: giga

    getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000  # time in ms

    results = []
    for i, chip in enumerate(SoC):
        chip_results = []
        for j, implementation in enumerate(implementations):
            raw = [getTime(chipPath[i], size, trial, implementation) for trial in range(1, 6)] 
            times = np.array(raw) / 1_000  # conversion: seconds
            flops = np.full(5, flop) / times

            mean = np.mean(flops, axis=0) 
            std = np.std(flops, axis=0)
            chip_results.append((mean, std))

        results.append(chip_results)

    bar_width = 0.3
    fig, axs = plt.subplots(len(SoC), 1, figsize=(6, 6), sharex=True)
    handles = []

    for i, (chip, ax) in enumerate(zip(SoC, axs)):
        means = [results[i][j][0] for j in range(len(implementations))]
        stds = [results[i][j][1] for j in range(len(implementations))]

        # Plot horizontal bar chart with hatches
        y_positions = np.arange(len(implementations))
        for j, (mean, std) in enumerate(zip(means, stds)):
            bar = ax.barh(
                y_positions[len(y_positions)-j - 1], mean, xerr=std, height=bar_width, alpha=0.8, 
                label=labels[j], hatch=formats[j+2], color=shifted_colors[j]
            )
            if i == 0:  # Collect legend handles only for the first subplot
                handles.append(bar)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        ax.set_title(f"{chip}", fontsize=10, loc="left")

        if i == len(SoC) - 1:
            ax.set_xlabel(f"GFLOPS-{size}x{size}", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        handles=[h[0] for h in handles],
        labels=labels,
        loc="upper right", ncol=1,  frameon=True)
    

    # Global adjustments
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/GFLOPS-{component}-ACC.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

def calculate_flops_allchips_best(chipPath, sizes):

    implementations = ["blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    labels = ["CPU-Accelerate", "GPU-Naive", "GPU-CUTLASS", "GPU-MPS"] 

    print("\n\nGFLOPS-all")

    # flop = num_flop(size) / 1_000_000_000  # conversion: giga

    getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000  # time in ms

    results = [] # different cjips
    for i, chip in enumerate(SoC):
        chip_results = [] # different implementations
        for j, implementation in enumerate(implementations):
            best = [] # best size
            for k, size in enumerate(sizes):
                flop = num_flop(size) / 1_000_000_000
                raw = [getTime(chipPath[i], size, trial, implementation) for trial in range(1, 6)] 
                times = np.array(raw) / 1_000  # conversion: seconds
                flops = np.full(5, flop) / times

                mean = np.mean(flops, axis=0) 
                std = np.std(flops, axis=0)

                if len(best) == 0 or mean > best[0]:
                    best = (mean, std, size)
                
            chip_results.append(best)

        results.append(chip_results)

    bar_width = 0.3
    fig, axs = plt.subplots(len(SoC), 1, figsize=(6, 6), sharex=True)
    handles = []

    for i, (chip, ax) in enumerate(zip(SoC, axs)):
        means = [results[i][j][0] for j in range(len(implementations))]
        stds = [results[i][j][1] for j in range(len(implementations))]
        best_sizes = [results[i][j][2] for j in range(len(implementations))]

        # Plot horizontal bar chart with hatches
        y_positions = np.arange(len(implementations))
        for j, (mean, std) in enumerate(zip(means, stds)):
            bar = ax.barh(
                y_positions[len(y_positions)-j - 1], mean, xerr=std, height=bar_width, alpha=0.8, 
                label=labels[j], hatch=formats[j+2], color=shifted_colors[j]
            )
            if i == 0:  # Collect legend handles only for the first subplot
                handles.append(bar)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(best_sizes)
        ax.set_title(f"{chip}", fontsize=10, loc="left")

        if i == len(SoC) - 1:
            ax.set_xlabel(f"GFLOPS-Best", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        handles=[h[0] for h in handles],
        labels=labels,
        loc="upper right", ncol=1,  frameon=True)
    

    # Global adjustments
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/GFLOPS-Best-acc.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def flopsAll(chipPath):
    implementations = ["baseline", "omp", "blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    labels = ["CPU-Single", "CPU-OMP", "CPU-Accelerate", "GPU-Naive", "GPU-CUTLASS", "GPU-MPS"]
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]

    print("\n\nGFLOPS-ALL")

    getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000  # time in ms

    results = [] # different chips
    for i, chip in enumerate(SoC):
        chip_results = [] # different implementations
        for j, implementation in enumerate(implementations):
            impl_results = [] # different sizes
            for k, size in enumerate(sizes):
                flop = num_flop(size) / 1_000_000_000
                raw = [] 
    
                try:
                    for trial in range(1, 6):
                        raw.append(getTime(chipPath[i], size, trial, implementation))
                except:
                    print(f"Warning: {chipPath[i]}/{size}x{size}/{trial}/{implementation}/timing.txt not found.")
                    continue
                times = np.array(raw) / 1_000  # conversion: seconds
                flops = np.full(5, flop) / times

                mean = np.mean(flops, axis=0) 
                std = np.std(flops, axis=0)

                impl_results.append((mean, std))
                
            chip_results.append(impl_results)

        results.append(chip_results)

    # have 4 sub plots panels, each panel is a chip, 2 rows, 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(18, 8), sharex=True, sharey=True)

    barWidth = 0.15

    for i, chip in enumerate(SoC):
        ax = axs[i // 2][i % 2]
        ax.set_title(f"{chip}", fontsize=12, loc="left")
        # ax.set_xlabel("Matrix Size")
        # ax.set_ylabel("GFLOPS")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)
        ax.set_yscale("log")

        for j, implementation in enumerate(implementations):
            dataNum = len(results[i][j])
            means = [results[i][j][k][0] for k in range(dataNum)]
            stds = [results[i][j][k][1] for k in range(dataNum)]
            x_offset = [i + (j +0.5 - len(implementations) / 2) * barWidth for i in range(dataNum)]  # 偏移使得每个 size 的 bars 排列整齐
            ax.bar(x_offset, means, width=barWidth, yerr=stds, capsize=3, label=labels[j], hatch=formats[j])

    fig.legend(labels=labels, loc="lower center", ncol=len(implementations), frameon=False)
    plt.subplots_adjust(left=0.12, bottom=0.12, top=0.88)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 1])
    fig.text(0.04, 0.5, "GFLOPS", va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.04, "Matrix Size", ha='center', fontsize=14)
    plt.savefig(f"{RESULTS_FOLDER}/GFLOPS-ALL.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


    



def removePowerWarmupInfo(chipPath):
    implementations = ["baseline", "omp", "blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    trial = 5
    for i, path in enumerate(chipPath):
        outPath = os.path.join(path, RAW_DATA_FOLDER)
        for j, size in enumerate(sizes):
            for k in range(1, trial + 1):
                for implementation in implementations:
                    powerFilePath = f"{outPath}/{size}x{size}/{k}/{implementation}/power.txt"
                    
                    if not os.path.exists(powerFilePath):
                        print(f"Warning: {powerFilePath} not found.")
                        continue
                    
                    sampleLine = []
                    with open(powerFilePath, "r") as f:
                        lines = f.readlines()
                        sample = 0
                        for i, line in enumerate(lines):
                            if "*** Sampled system activity" in line:
                                sampleLine.append(i)
                                sample += 1
                        if sample < 2:
                            continue


                    newLines = lines[sampleLine[1]:]

                    with open(powerFilePath, "w") as f:
                        f.writelines(newLines)





if __name__ == "__main__":
    # read argument 
    if len(sys.argv) < 2:
        print("Usage: python plot.py data path")
        sys.exit(1)
    path = sys.argv[1]

    chipPath = [os.path.join(path, chip) for chip in SoC]

    RESULTS_FOLDER = os.path.join(path, RESULTS_FOLDER)
    setup(RESULTS_FOLDER)

    removePowerWarmupInfo(chipPath)

    flopsAll(chipPath)
    plot_power_allchips_bar(chipPath, [2048, 4096, 8192, 16384])

    calculate_flops_allchips_acc(chipPath, 8192)
    calculate_flops_allchips_cpu(chipPath, 1024)
    calculate_flops_allchips_best(chipPath, [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    plot_power_allchips(chipPath, [2048, 4096, 8192, 16384])
    plot_efficiency_allchips(chipPath, [2048, 4096, 8192, 16384])

    



