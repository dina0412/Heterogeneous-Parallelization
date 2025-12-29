#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

/* ================================
   ЗАДАЧА 2. МАССИВЫ И OpenMP
   ================================ */

void minMaxOpenMP() {
    const int N = 10000;
    int* arr = new int[N];

    srand(time(nullptr));
    for (int i = 0; i < N; i++)
        arr[i] = rand() % 100000;

    auto startSeq = high_resolution_clock::now();
    int minSeq = arr[0], maxSeq = arr[0];
    for (int i = 1; i < N; i++) {
        if (arr[i] < minSeq) minSeq = arr[i];
        if (arr[i] > maxSeq) maxSeq = arr[i];
    }
    auto endSeq = high_resolution_clock::now();

    auto startPar = high_resolution_clock::now();
    int minPar = arr[0], maxPar = arr[0];

    #pragma omp parallel for reduction(min:minPar) reduction(max:maxPar)
    for (int i = 0; i < N; i++) {
        if (arr[i] < minPar) minPar = arr[i];
        if (arr[i] > maxPar) maxPar = arr[i];
    }
    auto endPar = high_resolution_clock::now();

    cout << "ЗАДАЧА 2\n";
    cout << "минимум: " << minPar << " максимум: " << maxPar << endl;
    cout << "последовательно: "
         << duration_cast<microseconds>(endSeq - startSeq).count() << " мкс\n";
    cout << "параллельно: "
         << duration_cast<microseconds>(endPar - startPar).count() << " мкс\n\n";

    delete[] arr;
}

/* ================================
   ЗАДАЧА 3. СОРТИРОВКА ВЫБОРОМ
   ================================ */

void selectionSort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[minIdx]) minIdx = j;
        swap(a[i], a[minIdx]);
    }
}

void selectionSortOMP(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            #pragma omp critical
            {
                if (a[j] < a[minIdx])
                    minIdx = j;
            }
        }
        swap(a[i], a[minIdx]);
    }
}

vector<int> generateArray(int n) {
    vector<int> a(n);
    for (int i = 0; i < n; i++)
        a[i] = rand() % 100000;
    return a;
}

void testSelectionSort() {
    vector<int> sizes = {1000, 10000};

    cout << "ЗАДАЧА 3\n";
    for (int n : sizes) {
        vector<int> data = generateArray(n);

        auto a1 = data;
        auto s1 = high_resolution_clock::now();
        selectionSort(a1);
        auto e1 = high_resolution_clock::now();

        auto a2 = data;
        auto s2 = high_resolution_clock::now();
        selectionSortOMP(a2);
        auto e2 = high_resolution_clock::now();

        cout << "размер массива: " << n << endl;
        cout << "последовательно: "
             << duration_cast<milliseconds>(e1 - s1).count() << " мс\n";
        cout << "параллельно: "
             << duration_cast<milliseconds>(e2 - s2).count() << " мс\n\n";
    }
}

/* ================================
   ЗАДАЧА 4. СОРТИРОВКА СЛИЯНИЕМ CUDA
   ================================ */

__device__ void merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left, j = mid, k = left;
    while (i < mid && j < right)
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
    while (i < mid) temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];
    for (i = left; i < right; i++)
        arr[i] = temp[i];
}

__global__ void mergeKernel(int* arr, int* temp, int n, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = idx * 2 * width;
    if (left < n) {
        int mid = min(left + width, n);
        int right = min(left + 2 * width, n);
        merge(arr, temp, left, mid, right);
    }
}

void mergeSortCUDA(int n) {
    int* h = new int[n];
    for (int i = 0; i < n; i++)
        h[i] = rand() % 100000;

    int *d, *temp;
    cudaMalloc(&d, n * sizeof(int));
    cudaMalloc(&temp, n * sizeof(int));
    cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice);

    auto start = high_resolution_clock::now();

    for (int width = 1; width < n; width *= 2) {
        int blocks = (n + 2 * width - 1) / (2 * width);
        mergeKernel<<<blocks, 1>>>(d, temp, n, width);
        cudaDeviceSynchronize();
    }

    auto end = high_resolution_clock::now();

    cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    cudaFree(temp);
    delete[] h;

    cout << "ЗАДАЧА 4\n";
    cout << "размер массива: " << n << endl;
    cout << "время GPU: "
         << duration_cast<milliseconds>(end - start).count() << " мс\n\n";
}

/* ================================
   MAIN
   ================================ */

int main() {
    minMaxOpenMP();
    testSelectionSort();
    mergeSortCUDA(10000);
    mergeSortCUDA(100000);
    return 0;
}
