/*
 * -> Solution for https://leetgpu.com/challenges/softmax-attention
 * -> Pointers:
 * --> https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
 * --> https://medium.com/data-science-collective/online-softmax-to-flash-attention-and-why-it-matters-9d676e7c50a8
 */

#include <cfloat>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

/**
 * Compute QK^T / sqrt(d)
 * @param Q Query - an M x d matrix
 * @param K Key - an N x d matrix
 * @param S Score matrix - an M x N matrix
 * @param M Number of rows in Q
 * @param N Number of rows in K
 * @param d Dimensionality of Q and K
 */
__global__ void compute_scores_kernel(const float* Q, const float* K, float* S, const int M, const int N, const int d) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only for elements inside the matrix
    if (row < M && col < N) {
        float sum = 0.0f;
        // Multiply the row'th-row of Q with the col'th column of K^T
        // But the col'th column of K^T is actually the col'th row of K.
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];
        }
        S[row * N + col] = sum / sqrtf(static_cast<float>(d));
    }
}

/**
 * Online Softmax kernel merges max/sum calculation and normalization.
 * @param S Score matrix (MxN)
 * @param A Attention weights output array (MxN)
 * @param M Number of rows in S
 * @param N Number of columns in S
 */
__global__ void online_softmax_kernel(const float* S, float* A, const int M, const int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float m = -FLT_MAX;
        float d = 0.0f;

        // Pass 1: Compute online max and sum
        for (int col = 0; col < N; col++) {
            float x = S[row * N + col];
            if (x > m) {
                d = d * expf(m - x) + 1.0f;
                m = x;
            } else {
                d += expf(x - m);
            }
        }

        // Pass 2: Apply normalization and write to global memory
        for (int col = 0; col < N; col++) {
            A[row * N + col] = expf(S[row * N + col] - m) / d;
        }
    }
}

/**
 * Naive matmul implementation C = A@B
 * @param A - A - MxN matrix
 * @param B - B - NxD matrix
 * @param C - Result - MxD matrix
 * @param M - Rows in A, rows in C
 * @param N - Columns in A, rows in B
 * @param D - Columns in B, columns in C
 */
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int D) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < D) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * D + col];
        }
        C[row * D + col] = sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    dim3 threadsPerBlock_2D(16, 16);
    dim3 blocksPerGrid_2D((N + threadsPerBlock_2D.x - 1) / threadsPerBlock_2D.x,
                       (M + threadsPerBlock_2D.y - 1) / threadsPerBlock_2D.y);

    float *d_score_matrix, *d_attention_weights;
    cudaMalloc(&d_score_matrix, M * N * sizeof(float));
    cudaMalloc(&d_attention_weights, M * N * sizeof(float));

    // Step 1: Compute QK^T / sqrt(d)
    compute_scores_kernel<<<blocksPerGrid_2D, threadsPerBlock_2D>>>(Q, K, d_score_matrix, M, N, d);

    // Step 2: Online Softmax
    int threadsPerBlock_1D = 256;
    int blocksPerGrid_1D = (M + threadsPerBlock_1D - 1) / threadsPerBlock_1D;
    online_softmax_kernel<<<blocksPerGrid_1D, threadsPerBlock_1D>>>(d_score_matrix, d_attention_weights, M, N);

    // Step 3: Final output A * V: M x d
    dim3 blocksPerGrid_out((d + threadsPerBlock_2D.x - 1) / threadsPerBlock_2D.x,
                           (M + threadsPerBlock_2D.y - 1) / threadsPerBlock_2D.y);
    matmul_naive_kernel<<<blocksPerGrid_out, threadsPerBlock_2D>>>(d_attention_weights, V, output, M, N, d);

    cudaDeviceSynchronize();

    cudaFree(d_score_matrix);
    cudaFree(d_attention_weights);
}

enum TestCaseType {
    TESTCASE_1,
    TESTCASE_2
};


void testcase(const TestCaseType type) {
    std::vector<float> h_Q, h_K, h_V, h_expected;
    int M, N, d;

    if (type == TESTCASE_1) {
        M = 2; N = 3; d = 4;
        h_Q = {1.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 1.0f, 0.0f, 0.0f};
        h_K = {1.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 1.0f, 0.0f, 0.0f,
               0.0f, 0.0f, 1.0f, 0.0f};
        h_V = {1.0f, 2.0f, 3.0f, 4.0f,
               5.0f, 6.0f, 7.0f, 8.0f,
               9.0f, 10.0f, 11.0f, 12.0f};
        h_expected = {4.29f, 5.29f, 6.29f, 7.29f,
                      5.00f, 6.00f, 7.00f, 8.00f};
    } else if (type == TESTCASE_2) {
        M = 1; N = 2; d = 2;
        h_Q = {1.0f, 2.0f};
        h_K = {1.0f, 0.0f,
               0.0f, 1.0f};
        h_V = {3.0f, 4.0f,
               5.0f, 6.0f};
        h_expected = {4.34f, 5.34f};
    } else {
        return;
    }

    std::vector<float> h_output(M * d);

    float *d_Q, *d_K, *d_V, *d_output;

    cudaMalloc(&d_Q, M * d * sizeof(float));
    cudaMalloc(&d_K, N * d * sizeof(float));
    cudaMalloc(&d_V, N * d * sizeof(float));
    cudaMalloc(&d_output, M * d * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), M * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);

    // Timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    solve(d_Q, d_K, d_V, d_output, M, N, d);

    // Timing stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output.data(), d_output, M * d * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_passed = true;
    for (size_t i = 0; i < h_expected.size(); ++i) {
        float epsilon = 0.01f;
        if (std::abs(h_output[i] - h_expected[i]) > epsilon) {
            std::cout << "Test failed at index " << i << "! Expected " << h_expected[i]
                      << ", got " << h_output[i] << " (diff: " << std::abs(h_output[i] - h_expected[i])
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
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
}


int main() {
    std::cout << "Running TESTCASE_1 (cold GPU):" << std::endl;
    testcase(TESTCASE_1);

    std::cout << "\nRunning TESTCASE_1:" << std::endl;
    testcase(TESTCASE_1);

    std::cout << "\nRunning TESTCASE_2:" << std::endl;
    testcase(TESTCASE_2);

    return 0;
}
