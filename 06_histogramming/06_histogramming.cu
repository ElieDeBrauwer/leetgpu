/**
 * https://leetgpu.com/challenges/histogramming
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

__global__ static void histogram_kernel(const int *input, int *output, int N, int num_bins) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Set up and initialize shared memory.
    extern __shared__ int shared_bins[];

    for ( auto idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        shared_bins[idx] = 0;
    }
    __syncthreads();

    // Add bins to shared memory using stride
    for (auto i = thread_id; i < N; i += stride) {
        int bin_index = input[i];
        if (bin_index < num_bins && bin_index >= 0) {
            atomicAdd(&shared_bins[bin_index], 1);
        }
    }

    __syncthreads();

    // Write back to global memory
    for ( auto idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        if (shared_bins[idx] > 0) {
            atomicAdd(&output[idx], shared_bins[idx]);
        }
    }
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    // Initialize output
    cudaMemset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));
    size_t shared_mem_size = num_bins * sizeof(int);

    // Obtain SM count, figure out ideal occupancy
    int device_id = 0;
    int num_sm = 0;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device_id);

    int blocks_per_sm = 0;
    int threads_per_block = 256;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, histogram_kernel, threads_per_block, shared_mem_size);
    int blocks_per_grid = num_sm * blocks_per_sm;

    histogram_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(input, histogram, N, num_bins);
}

namespace {
    enum TestCaseType {
        TESTCASE_1,
        TESTCASE_2,
        TESTCASE_3,
        TESTCASE_4
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
    } else if (type == TESTCASE_3 || type == TESTCASE_4) {
        N = 1024 * 1024 * 1024;
        num_bins = 1024;
        std::cout << "Filling array...." << std::flush;
        for (int i = 0; i < num_bins * num_bins; ++i) {
            for (int j = 0; j < num_bins; ++j) {
                h_input.push_back(j);
            }
        }
        std::cout << "done\n";
        h_expected = std::vector<int>(num_bins, num_bins * num_bins);

        if (type == TESTCASE_4) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::cout << "Shuffling array...." << std::flush;
            std::ranges::shuffle(h_input, g);
            std::cout << "done\n";
        }
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
    std::cout << "Running TESTCASE_4:" << std::endl;
    testcase(TESTCASE_4);
    return 0;
}
