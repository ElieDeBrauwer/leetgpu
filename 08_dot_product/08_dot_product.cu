/**
 * https://leetgpu.com/challenges/dot-product
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
constexpr float EPSILON = 1e-4f;


/**
 * @brief Performs an intra-warp inclusive reduction sum using shuffle instructions.
 * 
 * This function sums up the values across the threads within a single warp.
 * 
 * @param val The initial input value for the thread.
 * @return float The inclusive prefix sum of the elements in the warp.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Performs a block-level inclusive reduction sum by using warp reduction followed by shared memory accumulation.
 * 
 * This function reduces the partial sums calculated within each warp and then performs a final reduction across all warps in the block.
 * 
 * @param val The initial input value for the thread.
 * @return float The inclusive prefix sum of all elements processed by this block.
 */
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32]; // Shared memory for 32 warps (max 1024 threads)
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warpId = threadIdx.x / 32;

    // 1. Partial reduction inside each warp
    val = warp_reduce_sum(val);

    // 2. Write each warp's total sum to shared memory
    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    // 3. Final reduction of warp sums using the first warp
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (warpId == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * @brief Performs a parallel dot product calculation across all threads in the block and accumulates the result atomically.
 * 
 * This kernel computes $\sum_{i=0}^{N-1} A[i] \cdot B[i]$ by dividing the work among threads.
 * Each thread calculates a partial sum for its assigned elements, which is then reduced within the block using `block_reduce_sum`
 * and finally accumulated into the global result atomically.
 * 
 * @param A Pointer to the input vector A (global memory).
 * @param B Pointer to the input vector B (global memory).
 * @param result Pointer to a single float which will store the final dot product sum.
 * @param N The total number of elements for the dot product.
 */
__global__ void dot_product_kernel(const float *A, const float *B, float *result, int N) {
    float local_sum = 0;

    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < N; i += stride) {
        local_sum += A[i] * B[i];
    }

    const float block_sum = block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}

/**
 * Solves the dot product problem.
 * 
 * @param d_A Pointer to the input vector A
 * @param d_B Pointer to the input vector B
 * @param d_out Pointer to the float which will contain the dot product of d_A and d_B.
 * @param N The length of d_A and d_B.
 */
extern "C" void solve(const float *d_A, const float *d_B, float *d_out, int N) {
    constexpr int threads_per_block = 256;
    int blocks_per_grid = 1; // Default for small N

    if (N > threads_per_block) {
        // Large N: limit blocks to saturate SM occupancy without high atomic contention
        int max_blocks = 1024;
        blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
        if (blocks_per_grid > max_blocks) {
            blocks_per_grid = max_blocks;
        }
    }

    dot_product_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_out, N);
}

namespace {
    enum TestCaseType {
        TESTCASE_1,
        TESTCASE_2,
        TESTCASE_3
    };
}

/**
 * Executes a test case.
 * 
 * @param type The type of test case to run.
 */
static void testcase(TestCaseType type) {
    int N;
    std::vector<float> h_A, h_B;
    float h_output = 0;
    float h_expected;

    if (type == TESTCASE_1) {
        N = 4;
        h_A = {1, 2, 3, 4};
        h_B = {5, 6, 7, 8};
        h_expected = 70;
    } else if (type == TESTCASE_2) {
        N = 3;
        h_A = {0.5, 1.5, 2.5};
        h_B = {2.0, 3.0, 4.0};
        h_expected = 15.5;
    } else if (type == TESTCASE_3) {
        std::cout << "Creating input...." << std::flush;
        N = 100000000;
        h_expected = 0;
        float a_val = 0;
        float b_val = -1;
        for (int i = 0; i < N; ++i) {
            h_A.push_back(a_val);
            h_B.push_back(b_val);
            h_expected += a_val * b_val;
            a_val += 1.0;
            if (a_val > 10) {
                a_val = 0;
                b_val *= -1;
            }
        }
        std::cout << "done\n";
    } else {
        return;
    }

    float *d_A, *d_B, *d_result;
    cudaMalloc(&d_A, static_cast<size_t>(N) * sizeof(float));
    cudaMalloc(&d_B, static_cast<size_t>(N) * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_A, d_B, d_result, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_output, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    if (std::abs(h_output - h_expected) < EPSILON) {
        std::cout << "Test case passed!" << std::endl;
    } else {
        std::cout << "Test case failed!" << std::endl;
        std::cout << h_expected << "!=" << h_output << std::endl;
    }
    std::cout << "Solve duration: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
}

int main() {
    std::cout << "Running TESTCASE_1 (cold GPU):" << std::endl;
    testcase(TESTCASE_1);
    std::cout << "Running TESTCASE_1:" << std::endl;
    testcase(TESTCASE_1);
    std::cout << "Running TESTCASE_2:" << std::endl;
    testcase(TESTCASE_2);
    std::cout << "Running TESTCASE_3:" << std::endl;
    testcase(TESTCASE_3);
    return 0;
}
