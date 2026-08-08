/**
 * https://leetgpu.com/challenges/prefix-sum
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
constexpr float EPSILON = 1e-4f;


/**
 * Performs an intra-warp inclusive prefix sum using shuffle instructions.
 * 
 * @param val The input value to be summed within the warp.
 * @return The inclusive prefix sum for the current thread.
 */
__device__ float intra_warp_sum(float val) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        const float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x % 32) >= offset) {
            val += temp;
        }
    }
    return val;
}

/**
 * Prefix sum kernel performs inclusive prefix sum. It uses warp shuffles for
 * intra-warp phase and shared memory to pass warp totals across the block.
 * Assumes 1024 threads are active.
 * @param s_data Shared data of size 1024 of which the prefix sum should be created, results are stored in s_data.
 */
__device__ void prefix_sum_kernel(float *s_data) {
    __shared__ float s_warp_sums[32]; // 1024 threads, 32 threads per warp -> 32 warp sums needed

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp_id = tid / 32;

    // 1. Intra-warp sum
    float val = intra_warp_sum(s_data[tid]);

    // 2. Write total sum of each warp (last thread in warp) to shared memory
    if (lane == 31) {
        s_warp_sums[warp_id] = val;
    }
    __syncthreads();

    // 3. First warp scans all warp totals
    if (warp_id == 0) {
        float warp_val = (tid < (blockDim.x / 32)) ? s_warp_sums[tid] : 0.0f;
        warp_val = intra_warp_sum(warp_val);
        s_warp_sums[tid] = warp_val;
    }
    __syncthreads();

    // 4. Add scanned offset from preceding warps to thread's local sum
    if (warp_id > 0) {
        val += s_warp_sums[warp_id - 1];
    }

    // Write final scanned value back to shared memory
    s_data[tid] = val;
}


/**
 * Performs a block-level prefix sum by loading a tile into shared memory,
 * executing the kernel, and writing results back to global memory.
 * Also exports the block sum from the last thread.
 * 
 * @param input Pointer to the input global memory array.
 * @param output Pointer to the output global memory array.
 * @param block_sum Pointer to the array of block sums.
 * @param N The number of elements to process.
 */
__global__ static void prescan_local_tiles(const float *input, float *output, float *block_sum, int N) {
    __shared__ float s_data[1024]; // Shared memory of size blockdim.x

    unsigned int tid = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load global memory tile into shared memory
    s_data[tid] = (global_idx < N) ? input[global_idx] : 0.0f;
    __syncthreads();

    // Perform block-level prefix sum
    prefix_sum_kernel(s_data);
    __syncthreads();

    // Write result to global output
    if (global_idx < N) {
        output[global_idx] = s_data[tid];
    }

    // Last thread should export its sum as sum of this tile
    if (tid == blockDim.x - 1) {
        block_sum[blockIdx.x] = s_data[tid];
    }
}

/**
 * Create the prefix sum of the block sums.
 * @param block_sums - Arrays which need to be summed
 * @param N - Amount of entries in the array
 */
__global__ void scan_block_sums(float *block_sums, int N) {
    __shared__ float s_data[1024];
    const unsigned int tid = threadIdx.x;

    // Load input into shared memory
    s_data[tid] = (tid < N) ? block_sums[tid] : 0.0f;
    __syncthreads();

    // Scan block sums
    prefix_sum_kernel(s_data);
    __syncthreads();

    // Write shared memory to output
    if (tid < N) {
        block_sums[tid] = s_data[tid];
    }
}

/**
 * Adds the scanned block offsets to the output array.
 * 
 * @param output Pointer to the output global memory array.
 * @param block_sums Pointer to the array of block sums.
 * @param N The number of elements to process.
 */
__global__ void add_block_offsets(float *output, const float *block_sums, int N) {
    if (blockIdx.x == 0) return; // No offsets for the first block

    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float offset = block_sums[blockIdx.x - 1];
    if (global_idx < N) {
        output[global_idx] += offset;
    }
}

/**
 * Solves the prefix sum problem using a multi-pass block-based approach.
 * 
 * @param d_in Pointer to the input global memory array.
 * @param d_out Pointer to the output global memory array.
 * @param N The number of elements to process.
 */
extern "C" void solve(const float *d_in, float *d_out, int N) {
    constexpr int threads_per_block = 1024;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    // Global memory for block sums
    float *d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    // Pass 1: Scan local tiles and output block sums
    prescan_local_tiles<<<num_blocks, threads_per_block>>>(d_in, d_out, d_block_sums, N);

    // Pass 2: Scan the block sums array
    if (num_blocks <= threads_per_block) {
        scan_block_sums<<<1, threads_per_block>>>(d_block_sums, num_blocks);
    } else {
        // Too many, recurse ...
        solve(d_block_sums, d_block_sums, num_blocks);
    }

    // Pass 3: Add scanned offsets back to local tiles
    add_block_offsets<<<num_blocks, threads_per_block>>>(d_out, d_block_sums, N);

    cudaFree(d_block_sums);
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
    std::vector<float> h_input, h_output, h_expected;

    if (type == TESTCASE_1) {
        N = 4;
        h_input = {1, 2, 3, 4};
        h_expected = {1, 3, 6, 10};
    } else if (type == TESTCASE_2) {
        N = 5;
        h_input = {5, -2, 3, 1, -4};
        h_expected = {5, 3, 6, 7, 3};
    } else if (type == TESTCASE_3) {
        std::cout << "Creating input...." << std::flush;
        N = 100000000;
        float cum_sum = 0;
        float val = -50;
        for (int i = 0; i < N; ++i) {
            h_input.push_back(val);
            cum_sum += val;
            h_expected.push_back(cum_sum);
            val++;
            if (val > 50) {
                val = -50;
            }
        }
        std::cout << "done\n";
    } else {
        return;
    }

    h_output.assign(N, 0);

    float *d_input, *d_output;
    cudaMalloc(&d_input, static_cast<size_t>(N) * sizeof(int));
    cudaMalloc(&d_output, static_cast<size_t>(N) * sizeof(int));
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_input, d_output, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_passed = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_output[i] - h_expected[i]) > EPSILON) {
            std::cout << "Test failed at index " << i << "! Expected " << h_expected[i] << ", got " << h_output[i] <<
                    std::endl;
            all_passed = false;
            break;
        }
    }


    if (all_passed) {
        std::cout << "Test case passed!" << std::endl;
    } else {
        std::cout << "Test case failed!" << std::endl;
    }
    std::cout << "Solve duration: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
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
