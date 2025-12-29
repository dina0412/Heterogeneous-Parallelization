#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

int main() {
    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // ЗАДАНИЕ 1
    // Массив 50 000, среднее значение
    {
        const int N = 50000;
        int* arr = new int[N];

        long long sum = 0;
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 100 + 1; // [1; 100]
            sum += arr[i];
        }

        double average = static_cast<double>(sum) / N;
        std::cout << "Task 1: Average value = " << average << std::endl;

        delete[] arr;
    };

    // ЗАДАНИЕ 2
    // Последовательный поиск min / max (1 000 000)
    const int N2 = 1'000'000;
    int* arr2 = new int[N2];

    for (int i = 0; i < N2; i++) {
        arr2[i] = rand();
    }

    auto start_seq = std::chrono::high_resolution_clock::now();

    int min_seq = arr2[0];
    int max_seq = arr2[0];

    for (int i = 1; i < N2; i++) {
        if (arr2[i] < min_seq) min_seq = arr2[i];
        if (arr2[i] > max_seq) max_seq = arr2[i];
    }

    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_seq = end_seq - start_seq;

    std::cout << "Task 2 (Sequential):\n";
    std::cout << "Min = " << min_seq << ", Max = " << max_seq << std::endl;
    std::cout << "Time = " << time_seq.count() << " seconds\n";

    std::cout << "\n";

    // ЗАДАНИЕ 3
    // Параллельный поиск min / max (OpenMP)
    int min_par = arr2[0];
    int max_par = arr2[0];

    auto start_par = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N2; i++) {
        if (arr2[i] < min_par) min_par = arr2[i];
        if (arr2[i] > max_par) max_par = arr2[i];
    }

    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_par = end_par - start_par;

    std::cout << "Task 3 (Parallel OpenMP):\n";
    std::cout << "Min = " << min_par << ", Max = " << max_par << std::endl;
    std::cout << "Time = " << time_par.count() << " seconds\n";

    delete[] arr2;

    std::cout << "\n";

    // ЗАДАНИЕ 4
    // Среднее значение (5 000 000): sequential vs OpenMP
    const int N4 = 5'000'000;
    int* arr4 = new int[N4];

    for (int i = 0; i < N4; i++) {
        arr4[i] = rand() % 100 + 1;
    }

    // Последовательно
    auto start_avg_seq = std::chrono::high_resolution_clock::now();

    long long sum_seq4 = 0;
    for (int i = 0; i < N4; i++) {
        sum_seq4 += arr4[i];
    }
    double avg_seq4 = static_cast<double>(sum_seq4) / N4;

    auto end_avg_seq = std::chrono::high_resolution_clock::now();

    // Параллельно
    auto start_avg_par = std::chrono::high_resolution_clock::now();

    long long sum_par4 = 0;
    #pragma omp parallel for reduction(+:sum_par4)
    for (int i = 0; i < N4; i++) {
        sum_par4 += arr4[i];
    }
    double avg_par4 = static_cast<double>(sum_par4) / N4;

    auto end_avg_par = std::chrono::high_resolution_clock::now();

    std::cout << "Task 4:\n";
    std::cout << "Sequential average = " << avg_seq4 << std::endl;
    std::cout << "Parallel average   = " << avg_par4 << std::endl;

    std::cout << "Sequential time = "
              << std::chrono::duration<double>(end_avg_seq - start_avg_seq).count()
              << " seconds\n";

    std::cout << "Parallel time   = "
              << std::chrono::duration<double>(end_avg_par - start_avg_par).count()
              << " seconds\n";

    delete[] arr4;

    return 0;
}


