/**
 * https://leetgpu.com/challenges/2d-convolution
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define MAX_KERNEL_SIZE 1024
__constant__ float c_kernel[MAX_KERNEL_SIZE];

__global__ void conv2d(const float *input, float *output, const int input_rows, const int input_cols, const int kernel_rows, const int kernel_cols, const int output_rows, const int output_cols) {
    // Shared memory tiling
    extern __shared__ float s_tile[];

    const unsigned int ty = threadIdx.y;
    const unsigned int tx = threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + ty;
    const unsigned int col = blockIdx.x * blockDim.x + tx;

    const unsigned int shared_rows = blockDim.y + kernel_rows - 1;
    const unsigned int shared_cols = blockDim.x + kernel_cols - 1;

    // Load data into shared memory
    for (unsigned int i = ty; i < shared_rows; i += blockDim.y) {
        const unsigned int input_row = blockIdx.y * blockDim.y + i;
        for (unsigned int j = tx; j < shared_cols; j += blockDim.x) {
            const unsigned int input_col = blockIdx.x * blockDim.x + j;
            if (input_row < input_rows && input_col < input_cols) {
                s_tile[i * shared_cols + j] = input[input_row * input_cols + input_col];
            } else {
                s_tile[i * shared_cols + j] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Actual convolution on the tile
    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_rows; i++) {
            for (int j = 0; j < kernel_cols; j++) {
                sum += s_tile[(ty + i) * shared_cols + (tx + j)] * c_kernel[i * kernel_cols + j];
            }
        }
        output[row * output_cols + col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, const int input_rows,
                      const int input_cols, const int kernel_rows, const int kernel_cols) {
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;

    // Copy kernel to constant memory
    size_t kernel_size = kernel_rows * kernel_cols * sizeof(float);
    if (kernel_rows * kernel_cols <= MAX_KERNEL_SIZE) {
        cudaMemcpyToSymbol(c_kernel, kernel, kernel_size, 0, cudaMemcpyDeviceToDevice);
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    const unsigned int shared_rows = threadsPerBlock.y + kernel_rows - 1;
    const unsigned int shared_cols = threadsPerBlock.x + kernel_cols - 1;
    size_t shared_mem_size = shared_rows * shared_cols * sizeof(float);

    conv2d<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
        input, output, input_rows, input_cols, kernel_rows, kernel_cols,
        output_rows, output_cols
    );
    cudaDeviceSynchronize();
}


enum TestCaseType {
    TESTCASE_1,
    TESTCASE_2,
    TESTCASE_3
};


void testcase(const TestCaseType type) {
    std::vector<float> h_input, h_kernel, h_expected;
    int input_rows, input_cols, kernel_rows, kernel_cols;

    if (type == TESTCASE_1) {
        input_rows = 3; input_cols = 3; kernel_rows = 2; kernel_cols = 2;
        h_input = {1.0f, 2.0f, 3.0f,
                   4.0f, 5.0f, 6.0f,
                   7.0f, 8.0f, 9.0f};
        h_kernel = {0.0f, 1.0f,
                    1.0f, 0.0f};
        h_expected = {6.0f, 8.0f,
                      12.0f, 14.0f};
    } else if (type == TESTCASE_2) {
        input_rows = 4; input_cols = 4; kernel_rows = 1; kernel_cols = 3;
        h_input = {1.0f, 1.0f, 1.0f, 1.0f,
                   1.0f, 2.0f, 3.0f, 1.0f,
                   1.0f, 4.0f, 5.0f, 1.0f,
                   1.0f, 1.0f, 1.0f, 1.0f};
        h_kernel = {1.0f, 0.0f, 1.0f};
        h_expected = {2.0f, 2.0f,
                      4.0f, 3.0f,
                      6.0f, 5.0f,
                      2.0f, 2.0f};
    } else if (type == TESTCASE_3) {
        input_rows = 3072; input_cols = 3072; kernel_rows = 31; kernel_cols = 31;
        h_input.assign(input_rows * input_cols, 1.0f);
        h_kernel.assign(kernel_rows * kernel_cols, 1.0f);
        int out_r = input_rows - kernel_rows + 1;
        int out_c = input_cols - kernel_cols + 1;
        h_expected.assign(out_r * out_c, 961.0f);
    } else {
        return;
    }

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    std::vector<float> h_output(output_rows * output_cols);

    float *d_input, *d_kernel, *d_output;

    cudaMalloc(&d_input, input_rows * input_cols * sizeof(float));
    cudaMalloc(&d_kernel, kernel_rows * kernel_cols * sizeof(float));
    cudaMalloc(&d_output, output_rows * output_cols * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), input_rows * input_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_rows * kernel_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_input, d_kernel, d_output, input_rows, input_cols, kernel_rows, kernel_cols);

    // Timing stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output.data(), d_output, output_rows * output_cols * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_passed = true;
    for (size_t i = 0; i < h_expected.size(); ++i) {
        float epsilon = 0.01f;
        if (std::abs(h_output[i] - h_expected[i]) > epsilon) {
            std::cout << "Test failed at index " << i << "! Expected " << h_expected[i]
                      << ", got " << h_output[i] << " (diff: " << std::abs(h_output[i] - h_expected[i])
                      << ", epsilon: " << epsilon << ")" << std::endl;
            all_passed = false;
            break;
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
    cudaFree(d_kernel);
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
