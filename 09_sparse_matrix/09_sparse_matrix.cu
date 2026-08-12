/**
 * https://leetgpu.com/challenges/sparse-matrix-vector-multiplication
 */

#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <cmath>
constexpr float EPSILON = 1e-4f;

/**
 * @brief Counts the number of non-zero elements in each row of a dense matrix.
 * 
 * @param dense_matrix Pointer to the dense matrix of size M x N.
 * @param row_counts   Pointer to an array of size M to store the count of non-zeros for each row.
 * @param m            Number of rows in the matrix.
 * @param n            Number of columns in the matrix.
 */
__global__ void count_nnz_per_row_kernel(
    const float* __restrict__ dense_matrix,
    int* __restrict__ row_counts,
    int m, int n)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        int count = 0;
        auto row_start = row * n;
        for (int col = 0; col < n; ++col) {
            if (dense_matrix[row_start + col] != 0.0f) {
                count++;
            }
        }
        row_counts[row] = count;
    }
}

/**
 * @brief Scatters non-zero values and their column indices from a dense matrix into CSR format.
 * 
 * @param dense_matrix   Pointer to the dense matrix of size M x N.
 * @param row_offsets    Pointer to the CSR row offsets array (pre-scanned counts).
 * @param csr_values     Pointer to the CSR values array (output).
 * @param csr_col_indices Pointer to the CSR column indices array (output).
 * @param m              Number of rows in the matrix.
 * @param n              Number of columns in the matrix.
 */
__global__ void fill_csr_kernel(
    const float* __restrict__ dense_matrix,
    const int* __restrict__ row_offsets,
    float* __restrict__ csr_values,
    int* __restrict__ csr_col_indices,
    int m, int n)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        int write_idx = row_offsets[row];
        auto row_start = row * n;

        for (int col = 0; col < n; ++col) {
            if (float val = dense_matrix[row_start + col]; val != 0.0f) {
                csr_values[write_idx] = val;
                csr_col_indices[write_idx] = col;
                write_idx++;
            }
        }
    }
}

/**
 * @brief Performs Sparse Matrix-Vector Multiplication (SpMV) using the CSR format with warp-level parallelism.
 * 
 * This kernel uses warp shuffles to perform a parallel reduction for each row.
 * 
 * @param num_rows    Number of rows in the CSR matrix.
 * @param row_offsets Pointer to the CSR row offsets array.
 * @param col_indices Pointer to the CSR column indices array.
 * @param values      Pointer to the CSR non-zero values array.
 * @param x           Pointer to the input vector.
 * @param y           Pointer to the output vector.
 */
__global__ void spmv_csr_warp_kernel(
    const int num_rows,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    // Global warp ID across the grid
    auto warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    auto lane_id = threadIdx.x % 32; // Thread rank inside the warp (0 to 31)

    auto row = warp_id;

    if (row < num_rows) {
        int row_start = row_offsets[row];
        int row_end   = row_offsets[row + 1];

        float sum = 0.0f;

        // Threads in the warp collaborate to read and multiply non-zeros for 'row'
        for (auto i = row_start + lane_id; i < row_end; i += 32) {
            sum += values[i] * x[col_indices[i]];
        }

        // Intra-warp parallel reduction using register shuffles
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // The leader thread (lane 0) writes the final computed row sum to y
        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}


/**
 * Solves the sparse matrix multiplication problem.
 * 
 * @param d_A Sparse matrix A (MxN)
 * @param d_x Input vector B (Nx1)
 * @param d_y Pointer output vector (Mx1)
 * @param M   Number of rows in the input matrix (or rows in the output vector)
 * @param N   Number of columns in the input matrix (or rows in the input vector)
 * @param nnz Number of non zero elements
 */
extern "C" void solve(const float* d_A, const float* d_x, float* d_y, int M, int N, int nnz) {

    int threads_per_block = 256;
    int blocks = (M + threads_per_block - 1) / threads_per_block;

    int *d_csr_row_offsets;
    cudaMalloc(&d_csr_row_offsets, static_cast<size_t>(M + 1) * sizeof(int));
    cudaMemset(d_csr_row_offsets, 0, static_cast<size_t>(M + 1) * sizeof(int));

    // 1. Count non-zeros per row (store temporarily in row_offsets[0..m-1])
    count_nnz_per_row_kernel<<<blocks, threads_per_block>>>(
        d_A, d_csr_row_offsets, M, N
    );

    // 2. Perform Parallel Exclusive Prefix Sum to convert counts to offsets
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Request workspace size for CUB ExclusiveSum, by passing d_temp_storage as nullptr.
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_csr_row_offsets, d_csr_row_offsets, M + 1
    );

    // Now allocate the requested size
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Execute scan (transforms counts [2, 1, 3] -> offsets [0, 2, 3, 6])
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_csr_row_offsets, d_csr_row_offsets, M + 1
    );

    cudaFree(d_temp_storage);

    int * d_csr_col_indices;
    float * d_csr_values;

    cudaMalloc(&d_csr_col_indices, static_cast<size_t>(nnz) * sizeof(int));
    cudaMalloc(&d_csr_values, static_cast<size_t>(nnz) * sizeof(float));

    cudaMemset(d_csr_col_indices, 0, static_cast<size_t>(nnz) * sizeof(int));
    cudaMemset(d_csr_values, 0, static_cast<size_t>(nnz) * sizeof(float));

    // 3. Fill CSR values and column indices
    fill_csr_kernel<<<blocks, threads_per_block>>>(
        d_A, d_csr_row_offsets, d_csr_values, d_csr_col_indices, M, N
    );

    // 4. Sparse matrix multiply with CSR.
    threads_per_block = 256; // Must be a multiple of 32
    int warps_per_block = threads_per_block / 32;

    // Total blocks needed to cover all 'm' rows
    blocks = (M + warps_per_block - 1) / warps_per_block;

    spmv_csr_warp_kernel<<<blocks, threads_per_block>>>(
    M, d_csr_row_offsets, d_csr_col_indices, d_csr_values, d_x, d_y
    );

    cudaFree(d_csr_values);
    cudaFree(d_csr_col_indices);
    cudaFree(d_csr_row_offsets);
}

namespace {
    enum TestCaseType {
        TESTCASE_1,
        TESTCASE_2,
    };
}

/**
 * Executes a test case.
 * 
 * @param type The type of test case to run.
 */
static void testcase(TestCaseType type) {
    std::vector<float> h_x, h_y, h_expected;
    std::vector<float> h_A;
    int m = 0, n = 0, nnz = 0;

    if (type == TESTCASE_1) {
        m = 3;
        n = 4;
        h_A = {
            5.0, 0.0, 0.0, 1.0,
            0.0, 2.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 4.0,
        };
        h_x = {1.0, 2.0, 3.0, 4.0};
        h_expected = {9.0, 13.0, 16.0};
    } else if (type == TESTCASE_2) {
        m = 1000;
        n = 10000;
        h_A.resize(m * n,  0.0f);
        h_x.resize(n, 1.0f);
        for (int i = 0; i < m ; i++) {
            h_A[i * n + i] = static_cast<float>(i);
            h_expected.push_back(static_cast<float>(i));
        }
    } else {
        return;
    }

    nnz = static_cast<int>(std::ranges::count_if(h_A, [](float v) { return v != 0.0; }));
    h_y.assign(m, 0.0f);

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, static_cast<size_t>(n * m) * sizeof(float));
    cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(float));
    cudaMalloc(&d_y, static_cast<size_t>(m) * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), m * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_A, d_x, d_y, m, n, nnz);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_y.data(), d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

    if (std::ranges::equal(
            h_y,
            h_expected,
            [] (const float x, const float y) { return std::abs(x - y) < EPSILON;})) {
        std::cout << "Test case passed!" << std::endl;
    } else {
        std::cout << "Test case failed!" << std::endl;
    }
    std::cout << "Solve duration: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    std::cout << "Running TESTCASE_1 (cold GPU):" << std::endl;
    testcase(TESTCASE_1);
    std::cout << "Running TESTCASE_1:" << std::endl;
    testcase(TESTCASE_1);
    std::cout << "Running TESTCASE_2:" << std::endl;
    testcase(TESTCASE_2);
    return 0;
}
