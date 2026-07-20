/**
 * https://leetgpu.com/challenges/histogramming
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ static void histogram_kernel(const int *input, int *output, int N) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N) {
        int bin_index = input[thread_id];
        atomicAdd(&output[bin_index], 1);
    }
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    histogram_kernel<<<blocks_per_grid, threads_per_block>>>(input, histogram, N);
}

namespace {
    enum TestCaseType {
        TESTCASE_1,
        TESTCASE_2,
        TESTCASE_3
    };
}

static void testcase(TestCaseType type) {
    int N, num_bins;
    std::vector<int> h_input, h_output, h_expected;

    if (type == TESTCASE_1) {
        N = 5; num_bins = 3;
        h_input = {0, 1, 2, 1, 0};
        h_expected = {2, 2, 1};
    } else if (type == TESTCASE_2) {
        N = 4; num_bins = 3;
        h_input = {3, 3, 3, 3};
        h_expected = {0, 0, 0, 4, 0};
    } else if (type == TESTCASE_3) {
        N = 1024 * 1024 * 1024;
        num_bins = 1024;
        for (int i = 0; i < num_bins * num_bins; ++i) {
            for (int j = 0; j < num_bins; ++j) {
                h_input.push_back(j);
            }
        }
        h_expected = std::vector<int>(num_bins, num_bins * num_bins);
    } else {
        return;
    }

    h_output.assign(num_bins, 0);

    int *d_input, *d_histogram;
    cudaMalloc(&d_input, static_cast<size_t>(N) * sizeof(int));
    cudaMalloc(&d_histogram, static_cast<size_t>(num_bins) * sizeof(int));

    cudaMemcpy(d_input, h_input.data(), static_cast<size_t>(N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_input, d_histogram, N, num_bins);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output.data(), d_histogram, static_cast<size_t>(num_bins) * sizeof(int), cudaMemcpyDeviceToHost);

    bool all_passed = true;
    if (h_output.size() != h_expected.size()) {
        all_passed = false;
    } else {
        for (int i = 0; i < num_bins; ++i) {
            if (h_output[i] != h_expected[i]) {
                std::cout << "Test failed at index " << i << "! Expected " << h_expected[i] << ", got " << h_output[i] << std::endl;
                all_passed = false;
            }
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
    cudaFree(d_histogram);
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
