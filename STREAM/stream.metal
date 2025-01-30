#include <metal_stdlib>
using namespace metal;

kernel void copyKernel(device const float *a [[buffer(0)]],
                       device float *c [[buffer(1)]],
                       constant uint &count [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) {
        c[id] = a[id];
    }
}

kernel void scaleKernel(device const float *a [[buffer(0)]],
                        device float *c [[buffer(1)]],
                        constant float &scalar [[buffer(2)]],
                        constant uint &count [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id < count) {
        c[id] = scalar * a[id];
    }
}

kernel void addKernel(device const float *a [[buffer(0)]],
                      device const float *b [[buffer(1)]],
                      device float *c [[buffer(2)]],
                      constant uint &count [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) {
        c[id] = a[id] + b[id];
    }
}

kernel void triadKernel(device const float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        device float *c [[buffer(2)]],
                        constant float &scalar [[buffer(3)]],
                        constant uint &count [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id < count) {
        c[id] = a[id] + scalar * b[id];
    }
}