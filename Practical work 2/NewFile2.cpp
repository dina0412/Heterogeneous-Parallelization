#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

/* ================= последовательные сортировки ================= */

// пузырёк
void bubbleSort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1])
                swap(a[j], a[j + 1]);
        }
    }
}

// выбор
void selectionSort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[minIdx])
                minIdx = j;
        }
        swap(a[i], a[minIdx]);
    }
}

// вставки
void insertionSort(vector<int>& a) {
    int n = a.size();
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

/* ================= параллельные сортировки ================= */

// пузырёк (параллельный внешний цикл)
void bubbleSortOMP(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// выбор (параллельный поиск минимума)
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

// вставки (частично параллельная, ограниченный эффект)
void insertionSortOMP(vector<int>& a) {
    int n = a.size();
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

/* ================= тестирование ================= */

vector<int> generateArray(int n) {
    vector<int> a(n);
    for (int i = 0; i < n; i++)
        a[i] = rand() % 100000;
    return a;
}

void testSort(const string& name,
              void (*sortFunc)(vector<int>&),
              const vector<int>& original) {

    vector<int> a = original;
    auto start = high_resolution_clock::now();
    sortFunc(a);
    auto end = high_resolution_clock::now();

    cout << name << ": "
         << duration_cast<milliseconds>(end - start).count()
         << " мс\n";
}

int main() {
    srand(time(nullptr));

    vector<int> sizes = {1000, 10000, 100000};

    for (int n : sizes) {
        cout << "\nразмер массива: " << n << endl;
        vector<int> data = generateArray(n);

        cout << "последовательно:\n";
        testSort("  пузырёк", bubbleSort, data);
        testSort("  выбор", selectionSort, data);
        testSort("  вставки", insertionSort, data);

        cout << "параллельно (OpenMP):\n";
        testSort("  пузырёк OMP", bubbleSortOMP, data);
        testSort("  выбор OMP", selectionSortOMP, data);
        testSort("  вставки OMP", insertionSortOMP, data);
    }

    return 0;
}
