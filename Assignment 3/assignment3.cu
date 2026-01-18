#include <cuda_runtime.h>
#include <iostream>

#define N 1000000
#define MULT 2.0f

// ===============================
// ЗАДАНИЕ 1
// ===============================

// Только глобальная память
__global__ void multiply_global(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] *= MULT;
}

// Shared memory
__global__ void multiply_shared(float* data) {
    __shared__ float sh[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < N) {
        sh[tid] = data[idx];
        __syncthreads();
        sh[tid] *= MULT;
        __syncthreads();
        data[idx] = sh[tid];
    }
}

// ===============================
// ЗАДАНИЕ 2
// ===============================

__global__ void vector_add(float* a, float* b, float* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

// ===============================
// ЗАДАНИЕ 3
// ===============================

// Коалесцированный доступ
__global__ void coalesced(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] += 1.0f;
}

// Некоалесцированный доступ
__global__ void non_coalesced(float* data) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < N)
        data[idx] += 1.0f;
}

// ===============================
// Таймер
// ===============================

float measure(void (*kernel)(float*), float* d, dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<grid, block>>>(d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ===============================
// MAIN
// ===============================

int main() {
    float* h = new float[N];
    for (int i = 0; i < N; i++) h[i] = 1.0f;

    float *d, *a, *b, *c;
    cudaMalloc(&d, N * sizeof(float));
    cudaMalloc(&a, N * sizeof(float));
    cudaMalloc(&b, N * sizeof(float));
    cudaMalloc(&c, N * sizeof(float));

    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a, h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block256(256);
    dim3 grid256((N + 255) / 256);

    std::cout << "=============================\n";
    std::cout << "ЗАДАНИЕ 1: Умножение массива\n";

    float t_global = measure(multiply_global, d, grid256, block256);
    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);
    float t_shared = measure(multiply_shared, d, grid256, block256);

    std::cout << "Global memory: " << t_global << " ms\n";
    std::cout << "Shared memory: " << t_shared << " ms\n";

    std::cout << "\n=============================\n";
    std::cout << "ЗАДАНИЕ 2: Сложение массивов\n";

    for (int bs : {128, 256, 512}) {
        dim3 block(bs);
        dim3 grid((N + bs - 1) / bs);

        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);

        cudaEventRecord(s);
        vector_add<<<grid, block>>>(a, b, c);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float ms;
        cudaEventElapsedTime(&ms, s, e);
        std::cout << "Block size " << bs << ": " << ms << " ms\n";
    }

    std::cout << "\n=============================\n";
    std::cout << "ЗАДАНИЕ 3: Доступ к памяти\n";

    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);
    float t_coal = measure(coalesced, d, grid256, block256);
    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);
    float t_non = measure(non_coalesced, d, grid256, block256);

    std::cout << "Coalesced: " << t_coal << " ms\n";
    std::cout << "Non-coalesced: " << t_non << " ms\n";

    std::cout << "\n=============================\n";
    std::cout << "ЗАДАНИЕ 4: Оптимизация\n";

    dim3 badBlock(32);
    dim3 badGrid((N + 31) / 32);

    float t_bad = measure(coalesced, d, badGrid, badBlock);
    float t_good = measure(coalesced, d, grid256, block256);

    std::cout << "Неоптимально (32): " << t_bad << " ms\n";
    std::cout << "Оптимально (256): " << t_good << " ms\n";

    cudaFree(d); cudaFree(a); cudaFree(b); cudaFree(c);
    delete[] h;
    return 0;
}
