/*
 * -> Solution for https://leetgpu.com/challenges/softmax
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

// Robust atomicMax for floats using atomicCAS
__device__ __forceinline__ void atomicMaxFloat(float* addr, const float val) {
    auto addr_as_int = reinterpret_cast<int *>(addr);
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(__int_as_float(assumed), val)));
    } while (assumed != old);
}

/**
 * Max reduction kernel using shared memory.
 * @param input Input values
 * @param global_max Where the global maximum will be stored.
 * @param N Size of the input array.
 */
__global__ void reduce_max_kernel(const float* input, float* global_max, const int N) {
    extern __shared__ float shared_max[];
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    if (idx < N) {
        local_max = input[idx];
    }
    shared_max[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(global_max, shared_max[0]);
    }
}

/**
 * Sum reduction kernel using shared memory using the max trick:  sum(e^(input - global_max))
 * @param input Input floats
 * @param output sum(e^(input - global_max))
 * @param global_max Global maximum
 * @param N Size of the input array
 */
__global__ void reduce_sum_kernel(const float* input, float* output, const float* global_max, const int N) {
    extern __shared__ float smem[];
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    float max_val = *global_max;
    smem[tid] = (idx < N) ? expf(input[idx] - max_val): 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

/**
 * Softmax kernel calculates softmax using the max trick
 * @param input Input array
 * @param output Output array
 * @param global_max Global maximum
 * @param global_sum Global sum (including max trick)
 * @param N Size of the input array
 */
__global__ void softmax_kernel(const float* input, float* output, const float* global_max, const float* global_sum, const int N) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float max_val = *global_max;
    const float sum_val = *global_sum;
    if (idx < N) {
        output[idx] = expf(input[idx] - max_val) / sum_val;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_max_val;
    float* d_sum_val;
    cudaMalloc(&d_max_val, sizeof(float));
    cudaMalloc(&d_sum_val, sizeof(float));

    float h_max_val = -FLT_MAX;
    float h_sum_val = 0.0f;
    cudaMemcpy(d_max_val, &h_max_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum_val, &h_sum_val, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int smemSize = threadsPerBlock * sizeof(float);

    reduce_max_kernel<<<blocksPerGrid, threadsPerBlock, smemSize>>>(input, d_max_val, N);
    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock, smemSize>>>(input, d_sum_val, d_max_val, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_max_val, d_sum_val, N);

    cudaFree(d_max_val);
    cudaFree(d_sum_val);

    cudaDeviceSynchronize();
}


enum TestCaseType {
    TESTCASE_1,
    TESTCASE_2,
    TESTCASE_3
};


void testcase(const TestCaseType type) {
    std::vector<float> h_input;
    std::vector<float> expected_outputs;

    if (type == TESTCASE_1) {
        h_input = {1.0f, 2.0f, 3.0f};
        expected_outputs = {0.09003057f, 0.24472847f, 0.66524096f};
    } else if (type == TESTCASE_2) {
        h_input = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
        expected_outputs = {2.047e-09f, 3.038e-07f, 4.509e-5f, 6.693e-3f, 9.933e-01f};
    } else if (type == TESTCASE_3) {
        h_input.assign( 500000,-100);
        expected_outputs.assign( 500000, 0);
        h_input[0] = 100;
        expected_outputs[0] = 1.0000;
    }

    const size_t N = h_input.size();
    std::vector<float> h_output(N);

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_input, d_output, static_cast<int>(N));

    // Timing stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_passed = true;
    for (size_t i = 0; i < N; ++i) {
        float epsilon = std::abs(expected_outputs[i]) * 1e-4f;
        if (epsilon < 1e-7f) epsilon = 1e-7f; // Floor for very small numbers

        if (std::abs(h_output[i] - expected_outputs[i]) > epsilon) {
            std::cout << "Test failed at index " << i << "! Expected " << expected_outputs[i]
                      << ", got " << h_output[i] << " (diff: " << std::abs(h_output[i] - expected_outputs[i])
                      << ", epsilon: " << epsilon << ")" << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "Test ran fine!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
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

    std::cout << "\nRunning TESTCASE_1:" << std::endl;
    testcase(TESTCASE_1);

    std::cout << "\nRunning TESTCASE_2:" << std::endl;
    testcase(TESTCASE_2);

    std::cout << "\nRunning TESTCASE_3:" << std::endl;
    testcase(TESTCASE_3);

    return 0;
}
