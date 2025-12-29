#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

/* ВСПОМОГАТЕЛЬНОЕ*/

vector<int> generateArray(int n) {
    vector<int> a(n);
    for (int i = 0; i < n; i++)
        a[i] = rand() % 1000000;
    return a;
}

/* CPU СОРТИРОВКИ */

void cpuMergeSort(vector<int>& a) {
    sort(a.begin(), a.end());
}

void cpuQuickSort(vector<int>& a) {
    sort(a.begin(), a.end());
}

void cpuHeapSort(vector<int>& a) {
    make_heap(a.begin(), a.end());
    sort_heap(a.begin(), a.end());
}

/* CUDA MERGE SORT*/

__global__ void mergeKernel(int* data, int* temp, int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = idx * 2 * width;

    if (left < n) {
        int mid = min(left + width, n);
        int right = min(left + 2 * width, n);

        int i = left, j = mid, k = left;
        while (i < mid && j < right)
            temp[k++] = (data[i] < data[j]) ? data[i++] : data[j++];
        while (i < mid) temp[k++] = data[i++];
        while (j < right) temp[k++] = data[j++];

        for (int t = left; t < right; t++)
            data[t] = temp[t];
    }
}

void gpuMergeSort(vector<int>& a) {
    int n = a.size();
    int *d, *temp;

    cudaMalloc(&d, n * sizeof(int));
    cudaMalloc(&temp, n * sizeof(int));
    cudaMemcpy(d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    for (int width = 1; width < n; width *= 2) {
        int blocks = (n + 2 * width - 1) / (2 * width);
        mergeKernel<<<blocks, 1>>>(d, temp, width, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(a.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    cudaFree(temp);
}

/* CUDA QUICK SORT */

__global__ void quickKernel(int* data, int n) {
    int i = threadIdx.x;
    if (i >= n) return;

    for (int j = i + 1; j < n; j++) {
        if (data[i] > data[j]) {
            int tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

void gpuQuickSort(vector<int>& a) {
    int n = a.size();
    int* d;
    cudaMalloc(&d, n * sizeof(int));
    cudaMemcpy(d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    quickKernel<<<1, min(n, 1024)>>>(d, n);
    cudaDeviceSynchronize();

    cudaMemcpy(a.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

/*  CUDA HEAP SORT (ДЕМОНСТРАЦИОННАЯ) */

__global__ void heapKernel(int* data, int n) {
    int i = threadIdx.x;
    if (i < n / 2) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int largest = i;

        if (left < n && data[left] > data[largest])
            largest = left;
        if (right < n && data[right] > data[largest])
            largest = right;

        if (largest != i) {
            int tmp = data[i];
            data[i] = data[largest];
            data[largest] = tmp;
        }
    }
}

void gpuHeapSort(vector<int>& a) {
    int n = a.size();
    int* d;
    cudaMalloc(&d, n * sizeof(int));
    cudaMemcpy(d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < n; i++) {
        heapKernel<<<1, min(n, 1024)>>>(d, n - i);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(a.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

/* СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ */

void testAll(int n) {
    cout << "\nРазмер массива: " << n << endl;
    vector<int> base = generateArray(n);

    auto test = [&](auto func, string name) {
        vector<int> a = base;
        auto s = high_resolution_clock::now();
        func(a);
        auto e = high_resolution_clock::now();
        cout << name << ": "
             << duration_cast<milliseconds>(e - s).count()
             << " мс\n";
    };

    cout << "CPU:\n";
    test(cpuMergeSort, "Merge sort");
    test(cpuQuickSort, "Quick sort");
    test(cpuHeapSort, "Heap sort");

    cout << "GPU:\n";
    test(gpuMergeSort, "Merge sort CUDA");
    test(gpuQuickSort, "Quick sort CUDA");
    test(gpuHeapSort, "Heap sort CUDA");
}

int main() {
    srand(time(nullptr));

    testAll(10000);
    testAll(100000);
    // testAll(1000000);  // можно включить при наличии мощного GPU

    return 0;
}