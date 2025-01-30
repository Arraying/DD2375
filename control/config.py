import os

sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# sizes = [8192, 16384]

enablePureCPU = True # for large matrix, baseline and omp can be disabled by setting this to False

def get_path(n: int, idx: int):
    return os.path.join("..", "data", f"matrix-{n}-{idx}.float32")
