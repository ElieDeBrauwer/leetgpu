/*
 * -> Solution for https://leetgpu.com/challenges/reduction
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

__global__ void reduction_kernel_naive(const float* A, float* B, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(B, A[idx]);
    }
}

__global__ void reduction_kernel(const float* A, float* B, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? A[idx] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(B, sdata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int total = N;
    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int smemSize = threadsPerBlock * sizeof(float);

    reduction_kernel<<<blocksPerGrid, threadsPerBlock, smemSize>>>(input, output, N);

    cudaDeviceSynchronize();
}


enum TestCaseType {
    TESTCASE_1,
    TESTCASE_2,
    TESTCASE_3
};


void testcase(TestCaseType type) {
    std::vector<float> h_input;
    float expected_output = 0.0f;

    if (type == TESTCASE_1) {
        h_input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        expected_output = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
    } else if (type == TESTCASE_2) {
        h_input = {-2.5f, 1.5f, -1.0f, 2.0f};
        expected_output = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
    } else if (type == TESTCASE_3) {
        h_input.assign(4194304, 1.0f);
        expected_output = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
    }

    const size_t N = h_input.size();
    float h_output = 0.0f;

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    float zero = 0.0f;
    cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice);

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

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    if (h_output == expected_output) {
        std::cout << "Test ran fine!" << std::endl;
    } else {
        std::cout << "Test failed! Expected " << expected_output << ", got " << h_output << std::endl;
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
