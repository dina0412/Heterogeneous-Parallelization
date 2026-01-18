#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define MAX_BLOCK 256

// ===============================
// 1. Генерация данных
// ===============================

void generate_array(float* arr, int N) {
    for (int i = 0; i < N; i++)
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
}

// ===============================
// 2. РЕДУКЦИЯ
// ===============================

// a) Только global memory
__global__ void reduction_global(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        atomicAdd(output, input[idx]);
}

// b) Global + Shared memory
__global__ void reduction_shared(float* input, float* output, int N) {
    __shared__ float sh[MAX_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sh[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(output, sh[0]);
}

// ===============================
// 3. СОРТИРОВКА
// ===============================

// Bubble sort внутри блока (локальная память)
__global__ void bubble_sort(float* data, int N) {
    int start = blockIdx.x * blockDim.x;
    int end = min(start + blockDim.x, N);

    for (int i = start; i < end; i++) {
        for (int j = start; j < end - 1; j++) {
            if (data[j] > data[j + 1]) {
                float tmp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = tmp;
            }
        }
    }
}

// Слияние подмассивов с shared memory
__global__ void merge_blocks(float* data, float* temp, int width, int N) {
    __shared__ float sh[2 * MAX_BLOCK];

    int tid = threadIdx.x;
    int start = 2 * blockIdx.x * width;

    int mid = min(start + width, N);
    int end = min(start + 2 * width, N);

    if (start + tid < end)
        sh[tid] = data[start + tid];
    __syncthreads();

    int i = 0, j = mid - start, k = start;
    int left_size = mid - start;
    int right_size = end - mid;

    while (i < left_size && j < left_size + right_size) {
        temp[k++] = (sh[i] < sh[j]) ? sh[i++] : sh[j++];
    }
    while (i < left_size) temp[k++] = sh[i++];
    while (j < left_size + right_size) temp[k++] = sh[j++];
}

// ===============================
// 4. ИЗМЕРЕНИЕ ВРЕМЕНИ
// ===============================

float measure_reduction(void (*kernel)(float*, float*, int),
                        float* d_in, float* d_out, int N) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaMemset(d_out, 0, sizeof(float));

    int blocks = (N + MAX_BLOCK - 1) / MAX_BLOCK;

    cudaEventRecord(s);
    kernel<<<blocks, MAX_BLOCK>>>(d_in, d_out, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

// ===============================
// MAIN
// ===============================

int main() {
    int sizes[3] = {10000, 100000, 1000000};

    for (int s = 0; s < 3; s++) {
        int N = sizes[s];
        std::cout << "\nРазмер массива: " << N << "\n";

        float* h = new float[N];
        generate_array(h, N);

        float *d, *d_out, *temp;
        cudaMalloc(&d, N * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMalloc(&temp, N * sizeof(float));

        cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);

        // --- РЕДУКЦИЯ ---
        float t_global = measure_reduction(reduction_global, d, d_out, N);
        float t_shared = measure_reduction(reduction_shared, d, d_out, N);

        std::cout << "Редукция (global): " << t_global << " ms\n";
        std::cout << "Редукция (shared): " << t_shared << " ms\n";

        // --- СОРТИРОВКА ---
        cudaEvent_t srt_s, srt_e;
        cudaEventCreate(&srt_s);
        cudaEventCreate(&srt_e);

        int blocks = (N + MAX_BLOCK - 1) / MAX_BLOCK;

        cudaEventRecord(srt_s);
        bubble_sort<<<blocks, MAX_BLOCK>>>(d, N);

        for (int width = MAX_BLOCK; width < N; width *= 2) {
            merge_blocks<<<blocks / 2 + 1, MAX_BLOCK>>>(d, temp, width, N);
            cudaMemcpy(d, temp, N * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaEventRecord(srt_e);
        cudaEventSynchronize(srt_e);

        float t_sort;
        cudaEventElapsedTime(&t_sort, srt_s, srt_e);
        std::cout << "Сортировка + слияние: " << t_sort << " ms\n";

        cudaFree(d);
        cudaFree(d_out);
        cudaFree(temp);
        delete[] h;
    }
    return 0;
}
