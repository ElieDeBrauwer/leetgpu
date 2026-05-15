/**
 * https://leetgpu.com/challenges/3d-convolution
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define MAX_KERNEL_SIZE 1024
__constant__ float c_kernel[MAX_KERNEL_SIZE];

__global__ void conv3d(const float *input, float *output, 
                       const int input_depth, const int input_rows, const int input_cols, 
                       const int kernel_depth, const int kernel_rows, const int kernel_cols, 
                       const int output_depth, const int output_rows, const int output_cols) {
    
    extern __shared__ float s_tile[];

    const int tz = threadIdx.z;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int out_z = blockIdx.z * blockDim.z + tz;
    const int out_y = blockIdx.y * blockDim.y + ty;
    const int out_x = blockIdx.x * blockDim.x + tx;

    const int shared_depth = blockDim.z + kernel_depth - 1;
    const int shared_rows = blockDim.y + kernel_rows - 1;
    const int shared_cols = blockDim.x + kernel_cols - 1;

    // Load data into shared memory
    for (int i = tz; i < shared_depth; i += blockDim.z) {
        int in_z = blockIdx.z * blockDim.z + i;
        for (int j = ty; j < shared_rows; j += blockDim.y) {
            int in_y = blockIdx.y * blockDim.y + j;
            for (int k = tx; k < shared_cols; k += blockDim.x) {
                int in_x = blockIdx.x * blockDim.x + k;
                if (in_z < input_depth && in_y < input_rows && in_x < input_cols) {
                    s_tile[i * shared_rows * shared_cols + j * shared_cols + k] = input[in_z * input_rows * input_cols + in_y * input_cols + in_x];
                } else {
                    s_tile[i * shared_rows * shared_cols + j * shared_cols + k] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    if (out_z < output_depth && out_y < output_rows && out_x < output_cols) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_depth; i++) {
            for (int j = 0; j < kernel_rows; j++) {
                for (int k = 0; k < kernel_cols; k++) {
                    sum += s_tile[(tz + i) * shared_rows * shared_cols + (ty + j) * shared_cols + (tx + k)] * c_kernel[i * kernel_rows * kernel_cols + j * kernel_cols + k];
                }
            }
        }
        output[out_z * output_rows * output_cols + out_y * output_cols + out_x] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, 
                      int input_depth, int input_rows, int input_cols, 
                      int kernel_depth, int kernel_rows, int kernel_cols) {

    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    size_t kernel_vol = kernel_depth * kernel_rows * kernel_cols;
    cudaMemcpyToSymbol(c_kernel, kernel, kernel_vol * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    dim3 threadsPerBlock(8, 8, 8); // 512 threads
    dim3 blocksPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (output_depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

    int shared_depth = threadsPerBlock.z + kernel_depth - 1;
    int shared_rows = threadsPerBlock.y + kernel_rows - 1;
    int shared_cols = threadsPerBlock.x + kernel_cols - 1;
    size_t shared_mem_size = shared_depth * shared_rows * shared_cols * sizeof(float);

    conv3d<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
        input, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        output_depth, output_rows, output_cols
    );
    cudaDeviceSynchronize();
}

enum TestCaseType {
    TESTCASE_1,
    TESTCASE_2
};

void testcase(TestCaseType type) {
    int in_d, in_r, in_c, k_d, k_r, k_c;
    std::vector<float> h_input, h_kernel, h_expected;

    if (type == TESTCASE_1) {
        in_d = 3; in_r = 3; in_c = 3;
        k_d = 2; k_r = 3; k_c = 3;
        h_input = {
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27
        };
        h_kernel = {
            1, 0, 0, 1, 1, 1, 0, 0, 0,
            1, 1, 0, 1, 1, 0, 0, 0, 1
        };
        h_expected = {82, 163};
    } else if (type == TESTCASE_2) {
        in_d = 2; in_r = 2; in_c = 2;
        k_d = 2; k_r = 2; k_c = 2;
        h_input = {
            1, 2, 3, 4,
            5, 6, 7, 8
        };
        h_kernel = {
            1, 1, 1, 1,
            1, 1, 1, 1
        };
        h_expected = {36};
    } else {
        return;
    }

    int out_d = in_d - k_d + 1;
    int out_r = in_r - k_r + 1;
    int out_c = in_c - k_c + 1;
    int out_size = out_d * out_r * out_c;
    std::vector<float> h_output(out_size);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_kernel, h_kernel.size() * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), h_kernel.size() * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_input, d_kernel, d_output, in_d, in_r, in_c, k_d, k_r, k_c);

    cudaMemcpy(h_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_passed = true;
    for (int i = 0; i < out_size; ++i) {
        if (std::abs(h_output[i] - h_expected[i]) > 1e-3) {
            std::cout << "Test failed at index " << i << "! Expected " << h_expected[i] << ", got " << h_output[i] << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "Test case passed!" << std::endl;
    } else {
        std::cout << "Test case failed!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    std::cout << "Running TESTCASE_1:" << std::endl;
    testcase(TESTCASE_1);
    std::cout << "\nRunning TESTCASE_2:" << std::endl;
    testcase(TESTCASE_2);
    return 0;
}
