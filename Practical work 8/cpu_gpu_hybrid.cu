
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define N 1000000
#define BLOCK 256

// =======================================================
// GPU KERNEL
// =======================================================
__global__ void gpu_kernel(int* a, int offset, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (i < n)
        a[i] *= 2;
}

// =======================================================
// CPU (OpenMP)
// =======================================================
void cpu_process(std::vector<int>& a, int start, int end) {
    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        a[i] *= 2;
    }
}

// =======================================================
// MAIN
// =======================================================
int main() {
    std::vector<int> h(N, 1);

    // ---------------- CPU only ----------------
    auto c0 = std::chrono::high_resolution_clock::now();
    cpu_process(h, 0, N);
    auto c1 = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double, std::milli>(c1 - c0).count();

    // ---------------- GPU only ----------------
    int* d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    auto g0 = std::chrono::high_resolution_clock::now();
    gpu_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d, 0, N);
    cudaDeviceSynchronize();
    auto g1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h.data(), d, N * sizeof(int), cudaMemcpyDeviceToHost);

    double gpu_time =
        std::chrono::duration<double, std::milli>(g1 - g0).count();

    // ---------------- HYBRID ----------------
    cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    auto h0 = std::chrono::high_resolution_clock::now();

    // CPU part
    cpu_process(h, 0, N / 2);

    // GPU part
    gpu_kernel<<<(N / 2 + BLOCK - 1) / BLOCK, BLOCK>>>(d, N / 2, N);
    cudaDeviceSynchronize();

    auto h1 = std::chrono::high_resolution_clock::now();

    double hybrid_time =
        std::chrono::duration<double, std::milli>(h1 - h0).count();

    cudaFree(d);

    // ---------------- RESULTS ----------------
    std::cout << "Array size: " << N << "\n";
    std::cout << "CPU (OpenMP) time (ms): " << cpu_time << "\n";
    std::cout << "GPU (CUDA) time (ms): " << gpu_time << "\n";
    std::cout << "Hybrid CPU+GPU time (ms): " << hybrid_time << "\n";

    return 0;
}
