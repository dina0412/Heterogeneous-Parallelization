
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

// =======================================================
// PARAMETERS
// =======================================================
#define N_CPU 10000000
#define N_GPU (1<<24)
#define N_HYBRID 1000000
#define BLOCK 256

// =======================================================
// TASK 2 — CUDA KERNELS (MEMORY ACCESS)
// =======================================================
__global__ void coalesced_kernel(float* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_GPU)
        a[i] *= 2.0f;
}

__global__ void non_coalesced_kernel(float* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (i * 32) % N_GPU;
    if (i < N_GPU)
        a[idx] *= 2.0f;
}

// =======================================================
// TASK 3 — HYBRID GPU KERNEL
// =======================================================
__global__ void hybrid_kernel(int* a, int offset, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (i < n)
        a[i] *= 2;
}

// =======================================================
// MAIN
// =======================================================
int main() {

    // ===================================================
    // TASK 1 — OpenMP CPU ANALYSIS
    // ===================================================
    std::cout << "========== TASK 1: OpenMP CPU ==========\n";

    std::vector<double> cpu_data(N_CPU, 1.0);

    double t0 = omp_get_wtime();

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N_CPU; i++)
        sum += cpu_data[i];

    double mean = sum / N_CPU;

    double var = 0.0;
    #pragma omp parallel for reduction(+:var)
    for (int i = 0; i < N_CPU; i++)
        var += (cpu_data[i] - mean) * (cpu_data[i] - mean);

    var /= N_CPU;

    double t1 = omp_get_wtime();

    std::cout << "Mean: " << mean << "\n";
    std::cout << "Variance: " << var << "\n";
    std::cout << "CPU time (s): " << t1 - t0 << "\n\n";

    // ===================================================
    // TASK 2 — CUDA MEMORY ACCESS
    // ===================================================
    std::cout << "========== TASK 2: CUDA Memory Access ==========\n";

    float* d_gpu;
    cudaMalloc(&d_gpu, N_GPU * sizeof(float));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    cudaEventRecord(e0);
    coalesced_kernel<<<N_GPU / BLOCK, BLOCK>>>(d_gpu);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float t_coal;
    cudaEventElapsedTime(&t_coal, e0, e1);

    cudaEventRecord(e0);
    non_coalesced_kernel<<<N_GPU / BLOCK, BLOCK>>>(d_gpu);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float t_noncoal;
    cudaEventElapsedTime(&t_noncoal, e0, e1);

    std::cout << "Coalesced access time (ms): " << t_coal << "\n";
    std::cout << "Non-coalesced access time (ms): " << t_noncoal << "\n\n";

    cudaFree(d_gpu);

    // ===================================================
    // TASK 3 — HYBRID CPU + GPU (ASYNC)
    // ===================================================
    std::cout << "========== TASK 3: Hybrid CPU + GPU ==========\n";

    std::vector<int> h(N_HYBRID, 1);
    int* d;
    cudaMalloc(&d, N_HYBRID * sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double h0 = omp_get_wtime();

    // async copy + GPU
    cudaMemcpyAsync(d, h.data(), N_HYBRID * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    hybrid_kernel<<<(N_HYBRID/2 + BLOCK - 1)/BLOCK, BLOCK, 0, stream>>>(
        d, N_HYBRID/2, N_HYBRID);

    // CPU part
    #pragma omp parallel for
    for (int i = 0; i < N_HYBRID/2; i++)
        h[i] *= 2;

    cudaMemcpyAsync(h.data(), d, N_HYBRID * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    double h1 = omp_get_wtime();

    std::cout << "Hybrid async time (s): " << h1 - h0 << "\n";

    cudaFree(d);

    std::cout << "\nALL TASKS 1–3 COMPLETED SUCCESSFULLY\n";
    return 0;
}
