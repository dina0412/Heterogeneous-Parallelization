
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#define BLOCK 256

// =======================================================
// REDUCTION KERNEL (shared memory, DOUBLE)
// =======================================================
__global__ void reduce_kernel(const double* in, double* out, int n) {
    __shared__ double s[BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    s[tid] = (i < n) ? in[i] : 0.0;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step)
            s[tid] += s[tid + step];
        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x] = s[0];
}

// =======================================================
// CPU FUNCTIONS
// =======================================================
double cpu_sum(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x;
    return s;
}

void cpu_scan(std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); i++)
        a[i] += a[i - 1];
}

// =======================================================
// GPU REDUCTION (GPU + CPU FINAL STEP)
// =======================================================
double gpu_reduce(const std::vector<double>& h) {
    int n = h.size();

    double *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(double));
    cudaMemcpy(d_in, h.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK - 1) / BLOCK;
    cudaMalloc(&d_out, blocks * sizeof(double));

    reduce_kernel<<<blocks, BLOCK>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    std::vector<double> partial(blocks);
    cudaMemcpy(partial.data(), d_out,
               blocks * sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return cpu_sum(partial);
}

// =======================================================
// PREFIX SCAN KERNEL
// =======================================================
__global__ void scan_kernel(const int* in, int* out, int n) {
    __shared__ int s[BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    s[tid] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int v = (tid >= offset) ? s[tid - offset] : 0;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }

    if (i < n)
        out[i] = s[tid];
}

// =======================================================
// TEST FUNCTION
// =======================================================
void run_test(int N) {
    std::cout << "\n=== N = " << N << " ===\n";

    // ---------- REDUCTION ----------
    std::vector<double> h(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; i++) h[i] = dist(gen);

    auto c0 = std::chrono::high_resolution_clock::now();
    double cpu_res = cpu_sum(h);
    auto c1 = std::chrono::high_resolution_clock::now();

    auto g0 = std::chrono::high_resolution_clock::now();
    double gpu_res = gpu_reduce(h);
    auto g1 = std::chrono::high_resolution_clock::now();

    std::cout << "Reduction CPU sum: " << cpu_res << "\n";
    std::cout << "Reduction GPU sum: " << gpu_res << "\n";
    std::cout << "CPU time (ms): "
              << std::chrono::duration<double, std::milli>(c1 - c0).count() << "\n";
    std::cout << "GPU time (ms): "
              << std::chrono::duration<double, std::milli>(g1 - g0).count() << "\n";

    // ---------- SCAN ----------
    std::vector<int> h_scan(N, 1);
    std::vector<int> h_scan_cpu = h_scan;

    c0 = std::chrono::high_resolution_clock::now();
    cpu_scan(h_scan_cpu);
    c1 = std::chrono::high_resolution_clock::now();

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_scan.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    g0 = std::chrono::high_resolution_clock::now();
    scan_kernel<<<N / BLOCK, BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    g1 = std::chrono::high_resolution_clock::now();

    std::cout << "Scan CPU time (ms): "
              << std::chrono::duration<double, std::milli>(c1 - c0).count() << "\n";
    std::cout << "Scan GPU time (ms): "
              << std::chrono::duration<double, std::milli>(g1 - g0).count() << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
}

// =======================================================
// MAIN
// =======================================================
int main() {
    run_test(1024);
    run_test(1'000'000);
    run_test(10'000'000);
    return 0;
}
