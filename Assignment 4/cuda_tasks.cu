
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N_SUM 100000
#define N_SCAN 1000000
#define BLOCK 256

// =======================================================
// TASK 1: SUM (global memory + block reduction)
// =======================================================
__global__ void sum_kernel(const float* a, float* res, int n) {
    __shared__ float buf[BLOCK];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    for (int i = gid; i < n; i += stride)
        local += a[i];

    buf[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(res, buf[0]);
}

float cpu_sum(const std::vector<float>& a) {
    float s = 0.0f;
    for (float x : a) s += x;
    return s;
}

void task1() {
    std::vector<float> h(N_SUM, 1.0f);

    auto c0 = std::chrono::high_resolution_clock::now();
    float s_cpu = cpu_sum(h);
    auto c1 = std::chrono::high_resolution_clock::now();

    float *d_a, *d_res, s_gpu = 0.0f;
    cudaMalloc(&d_a, N_SUM * sizeof(float));
    cudaMalloc(&d_res, sizeof(float));
    cudaMemcpy(d_a, h.data(), N_SUM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, sizeof(float));

    auto g0 = std::chrono::high_resolution_clock::now();
    sum_kernel<<<128, BLOCK>>>(d_a, d_res, N_SUM);
    cudaDeviceSynchronize();
    auto g1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(&s_gpu, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nTASK 1\nCPU sum=" << s_cpu
              << "\nGPU sum=" << s_gpu
              << "\nCPU time(ms)="
              << std::chrono::duration<double, std::milli>(c1 - c0).count()
              << "\nGPU time(ms)="
              << std::chrono::duration<double, std::milli>(g1 - g0).count()
              << "\n";

    cudaFree(d_a);
    cudaFree(d_res);
}

// =======================================================
// TASK 2: PREFIX SCAN (shared memory)
// =======================================================
__global__ void scan_kernel(const int* in, int* out, int n) {
    __shared__ int s[BLOCK];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    s[t] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (int off = 1; off < blockDim.x; off <<= 1) {
        int v = (t >= off) ? s[t - off] : 0;
        __syncthreads();
        s[t] += v;
        __syncthreads();
    }

    if (i < n) out[i] = s[t];
}

void task2() {
    std::vector<int> h(N_SCAN, 1);

    auto c0 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < N_SCAN; i++) h[i] += h[i - 1];
    auto c1 = std::chrono::high_resolution_clock::now();

    int *d_in, *d_out;
    cudaMalloc(&d_in, N_SCAN * sizeof(int));
    cudaMalloc(&d_out, N_SCAN * sizeof(int));
    cudaMemcpy(d_in, h.data(), N_SCAN * sizeof(int), cudaMemcpyHostToDevice);

    auto g0 = std::chrono::high_resolution_clock::now();
    scan_kernel<<<N_SCAN / BLOCK, BLOCK>>>(d_in, d_out, N_SCAN);
    cudaDeviceSynchronize();
    auto g1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nTASK 2\nCPU time(ms)="
              << std::chrono::duration<double, std::milli>(c1 - c0).count()
              << "\nGPU time(ms)="
              << std::chrono::duration<double, std::milli>(g1 - g0).count()
              << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
}

// =======================================================
// TASK 3: HYBRID CPU + GPU
// =======================================================
__global__ void gpu_work(int* a, int start, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i < n) a[i] *= 2;
}

void task3() {
    std::vector<int> h(N_SCAN, 1);
    int* d;
    cudaMalloc(&d, N_SCAN * sizeof(int));

    auto c0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_SCAN; i++) h[i] *= 2;
    auto c1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d, h.data(), N_SCAN * sizeof(int), cudaMemcpyHostToDevice);
    auto g0 = std::chrono::high_resolution_clock::now();
    gpu_work<<<(N_SCAN + BLOCK - 1) / BLOCK, BLOCK>>>(d, 0, N_SCAN);
    cudaDeviceSynchronize();
    auto g1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d, h.data(), N_SCAN * sizeof(int), cudaMemcpyHostToDevice);
    auto h0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_SCAN / 2; i++) h[i] *= 2;
    gpu_work<<<(N_SCAN / 2 + BLOCK - 1) / BLOCK, BLOCK>>>(d, N_SCAN / 2, N_SCAN);
    cudaDeviceSynchronize();
    auto h1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nTASK 3\nCPU time(ms)="
              << std::chrono::duration<double, std::milli>(c1 - c0).count()
              << "\nGPU time(ms)="
              << std::chrono::duration<double, std::milli>(g1 - g0).count()
              << "\nHybrid time(ms)="
              << std::chrono::duration<double, std::milli>(h1 - h0).count()
              << "\n";

    cudaFree(d);
}

// =======================================================
// MAIN
// =======================================================
int main() {
    task1();
    task2();
    task3();
    return 0;
}
