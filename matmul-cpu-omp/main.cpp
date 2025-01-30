#include <omp.h>
#include "testlib.h"

// Block size taken from: https://stackoverflow.com/a/54637578/14198415
// Will take 64KB as it is the lowest common denominator.
// Maximize 3 * x * x * sizeof(float) <= 64KB, should be ~50.
constexpr unsigned int BLOCK_SIZE = 50;

void block_parallel(unsigned int n, [[maybe_unused]] unsigned int memory_length, const float* left, const float* right, float* out) {
    unsigned int size = n;
    int i, j, k, jj, kk;
    float tmp;
    int chunk = 1;
    int tid;
#pragma omp parallel default(none) shared(n, left, right, out, size, chunk) private(i, j, k, jj, kk, tid, tmp)
    {
#pragma omp for schedule (static, chunk)
        for (jj = 0; jj < size; jj += BLOCK_SIZE)
        {
            for (kk = 0; kk < size; kk += BLOCK_SIZE)
            {
                for (i = 0; i < size; i++)
                {
                    for (j = jj; j < ((jj + BLOCK_SIZE) > size ? size : (jj + BLOCK_SIZE)); j++)
                    {
                        tmp = 0.0f;
                        for (k = kk; k < ((kk + BLOCK_SIZE) > size ? size : (kk + BLOCK_SIZE)); k++)
                        {
                            tmp += left[INDEX(n, i, k)] * right[INDEX(n, k, j)];
                        }
                        out[INDEX(n, i, j)] += tmp;
                    }
                }
            }
        }
    }
}

int main() {
    int numCores = omp_get_num_procs();
    omp_set_num_threads(numCores);
    test_suite(block_parallel);
}