/**
 * https://leetgpu.com/challenges/general-matrix-multiplication-gemm
 */

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <vector>
constexpr float EPSILON = 1e-4f;

using namespace nvcuda::wmma;
namespace {
    // Standard Tensor Core tile shape: 16x16x16
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    /**
     * CUDA kernel performs a GEMM (General Matrix Multiply) where C = alpha * ( A x B ) + beta * C using WMMA (warp
     * matrix multiply accumulate)
     *
     * @param A     Input matrix A (MxK)
     * @param B     Input matrix B (KxN)
     * @param C     Input/Output Matrix C (MxN)
     * @param M     Number of rows in A and C
     * @param N     Number of columns in B and C
     * @param K     Number of columns in A and number of rows in B
     * @param alpha Alpha value to be used in GEMM
     * @param beta  Beta value to be used in GEMM
     */
    __global__ void wmma_gemm_fp32_acc_kernel(
        const half *A, const half *B, half *C,
        const int M, const int N, const int K,
        float alpha, float beta)
    {
        // Flatten thread index inside 2D block
        const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const unsigned int laneId = tid % warpSize;
        const unsigned int warpIdInBlock = tid / warpSize;

        // 2x2 grid of warps per block (4 warps per block)
        const unsigned int warpRowInBlock = warpIdInBlock / 2;
        const unsigned int warpColInBlock = warpIdInBlock % 2;

        const unsigned int warpRow = blockIdx.y * 2 + warpRowInBlock;
        const unsigned int warpCol = blockIdx.x * 2 + warpColInBlock;

        const unsigned int tileRow = warpRow * WMMA_M;
        const unsigned int tileCol = warpCol * WMMA_N;

        // Early exit if the entire warp tile is outside matrix C boundaries
        if (tileRow >= M || tileCol >= N) return;

        // Shared memory staging tiles per warp for safe handling of arbitrary M, N, K
        __shared__ half s_A[4][WMMA_M * WMMA_K];
        __shared__ half s_B[4][WMMA_K * WMMA_N];
        __shared__ half s_C[4][WMMA_M * WMMA_N];

        // Declare Fragments
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        fill_fragment(c_frag, 0.0f);

        // Main Accumulation Loop along K dimension
        for (int k = 0; k < K; k += WMMA_K) {
            // Cooperatively load tile A into s_A (256 elements per warp tile, 32 threads in warp -> 8 elements/thread)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned int elemIdx = laneId * 8 + i;
                const unsigned int r = elemIdx / WMMA_K;
                const unsigned int c = elemIdx % WMMA_K;
                const unsigned int gr = tileRow + r;
                const unsigned int gc = k + c;
                s_A[warpIdInBlock][elemIdx] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
            }

            // Cooperatively load tile B into s_B
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned int elemIdx = laneId * 8 + i;
                const unsigned int r = elemIdx / WMMA_N;
                const unsigned int c = elemIdx % WMMA_N;
                const unsigned int gr = k + r;
                const unsigned int gc = tileCol + c;
                s_B[warpIdInBlock][elemIdx] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
            }

            __syncwarp();

            // Load FP16 sub-tiles from shared memory with stride WMMA_K / WMMA_N
            load_matrix_sync(a_frag, s_A[warpIdInBlock], WMMA_K);
            load_matrix_sync(b_frag, s_B[warpIdInBlock], WMMA_N);

            // Execute Tensor Core multiply-accumulate
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Scale and Write Back Output
        // Cooperatively load existing C tile into s_C for beta scaling (or 0 if beta == 0)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int elemIdx = laneId * 8 + i;
            const unsigned int r = elemIdx / WMMA_N;
            const unsigned int c = elemIdx % WMMA_N;
            const unsigned int gr = tileRow + r;
            const unsigned int gc = tileCol + c;
            s_C[warpIdInBlock][elemIdx] = (beta != 0.0f && gr < M && gc < N) ? C[gr * N + gc] : __float2half(0.0f);
        }

        __syncwarp();

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_orig_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_out_frag;

        load_matrix_sync(c_orig_frag, s_C[warpIdInBlock], WMMA_N, mem_row_major);

        for (int i = 0; i < nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float, void>::num_elements; i++) {
            float c_orig_fp32 = __half2float(c_orig_frag.x[i]);
            float result_fp32 = alpha * c_frag.x[i] + beta * c_orig_fp32;
            c_out_frag.x[i]   = __float2half(result_fp32);
        }

        store_matrix_sync(s_C[warpIdInBlock], c_out_frag, WMMA_N, mem_row_major);

        __syncwarp();

        // Write back valid elements from s_C to global memory C
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int elemIdx = laneId * 8 + i;
            const unsigned int r = elemIdx / WMMA_N;
            const unsigned int c = elemIdx % WMMA_N;
            const unsigned int gr = tileRow + r;
            const unsigned int gc = tileCol + c;
            if (gr < M && gc < N) {
                C[gr * N + gc] = s_C[warpIdInBlock][elemIdx];
            }
        }
    }
}
/**
 * Performs a GEMM (General Matrix Multiply) where C = alpha * ( A x B ) + beta * C
 * 
 * @param d_A   Input matrix A (MxK)
 * @param d_B   Input matrix B (KxN)
 * @param d_C   Input/Output Matrix C (MxN)
 * @param M     Number of rows in A and C
 * @param N     Number of columns in B and C
 * @param K     Number of columns in A and number of rows in B
 * @param alpha Alpha value to be used in GEMM
 * @param beta  Beta value to be used in GEMM
 */
extern "C" void solve(const half* d_A, const half* d_B, half* d_C, int M, int N, int K, float alpha, float beta) {

    // -------------------------------------------------------------
    // LAUNCH CONFIGURATION
    // -------------------------------------------------------------
    // 1. Each warp processes one (16x16) output tile.
    // 2. We use 4 warps (128 threads) per block as an example: 2 warps in X, 2 in Y.
    dim3 blockDim(64, 2); // 64 * 2 = 128 threads per block = 4 warps total

    // 3. Grid dimensions: divide total matrix tiles by block tile dimensions
    // One block covers (2 warps * 16 = 32) rows and (2 warps * 16 = 32) cols
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    // Launch the CUDA kernel
    wmma_gemm_fp32_acc_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
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
    std::vector<half> h_A, h_B, h_C, h_expected;
    int m = 0, n = 0, k = 0;
    float alpha = 0;
    float beta = 0;

    if (type == TESTCASE_1) {
        k = 3;
        m = 2;
        n = 2;
        h_A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        h_B = {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        };
        h_C = {
            1.0, 1.0,
            1.0, 1.0
        };
        h_expected = {
            22.0, 28.0,
            49.0, 64.0
        };
        alpha = 1.0;
        beta = 0.0;
    } else if (type == TESTCASE_2) {
        k = 4096;
        m = 4096;
        n = 4096;
        // m * k matrix filled with 2s
        h_A.assign(m * k, 2);
        // k * n unit matrix
        h_B.assign(k * n, 0);
        for (int i = 0; i < k; i++) {
            h_B[i * n + i] = 1.0;
        }
        // m * n matrix filled with 3s
        h_C.assign(m * n, 3);
        alpha = 2.0;
        beta = 3.0;

        // 2 * [2 matrix] * eye + 3 * [3 matrix] = [4 matrix] + [9 matrix] = [13 matrix]
        h_expected.assign(m * n, 13);
    } else {
        return;
    }


    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, static_cast<size_t>(m * k) * sizeof(half));
    cudaMalloc(&d_B, static_cast<size_t>(k * n) * sizeof(half));
    cudaMalloc(&d_C, static_cast<size_t>(m * n) * sizeof(half));
    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), m * n * sizeof(half), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_A, d_B, d_C, m, n, k, alpha, beta);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C.data(), d_C, m * n * sizeof(half), cudaMemcpyDeviceToHost);

    if (std::ranges::equal(
            h_C,
            h_expected,
            [] (const half x, const half y) { return std::abs(static_cast<float>(x - y)) < EPSILON;})) {
        std::cout << "Test case passed!" << std::endl;
    } else {
        std::cout << "Test case failed!" << std::endl;
    }
    std::cout << "Solve duration: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
