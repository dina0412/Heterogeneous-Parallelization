#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace chrono;

/* ЧАСТЬ 1. РАБОТА С МАССИВАМИ */

void part1_arrays() {

    // a) создание массива и заполнение случайными числами
    const int N = 30;
    int arr[N];

    srand(time(nullptr));
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100 + 1;
    }

    // c) вывод массива
    cout << "ЧАСТЬ 1. МАССИВ\n";
    cout << "массив:\n";
    for (int i = 0; i < N; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // b) последовательный поиск минимума и максимума
    auto start_seq = high_resolution_clock::now();

    int minSeq = arr[0];
    int maxSeq = arr[0];

    for (int i = 1; i < N; i++) {
        if (arr[i] < minSeq) minSeq = arr[i];
        if (arr[i] > maxSeq) maxSeq = arr[i];
    }

    auto end_seq = high_resolution_clock::now();

    // 2.a) параллельный поиск минимума и максимума
    auto start_par = high_resolution_clock::now();

    int minPar = arr[0];
    int maxPar = arr[0];

    #pragma omp parallel for reduction(min:minPar) reduction(max:maxPar)
    for (int i = 0; i < N; i++) {
        if (arr[i] < minPar) minPar = arr[i];
        if (arr[i] > maxPar) maxPar = arr[i];
    }

    auto end_par = high_resolution_clock::now();

    // c) вывод минимума и максимума
    cout << "минимум: " << minPar << endl;
    cout << "максимум: " << maxPar << endl;

    // 2.b) сравнение времени выполнения
    cout << "время (последовательно): "
         << duration_cast<microseconds>(end_seq - start_seq).count()
         << " мкс\n";

    cout << "время (параллельно): "
         << duration_cast<microseconds>(end_par - start_par).count()
         << " мкс\n\n";
}

/* ЧАСТЬ 2. СТРУКТУРЫ ДАННЫХ */

struct Node {
    int data;
    Node* next;
};

// 1.a) односвязный список

void list_add(Node*& head, int value) {
    Node* newNode = new Node{value, head};
    head = newNode;
}

bool list_find(Node* head, int value) {
    while (head) {
        if (head->data == value) return true;
        head = head->next;
    }
    return false;
}

void list_remove(Node*& head, int value) {
    if (!head) return;

    if (head->data == value) {
        Node* tmp = head;
        head = head->next;
        delete tmp;
        return;
    }

    Node* curr = head;
    while (curr->next && curr->next->data != value) {
        curr = curr->next;
    }

    if (curr->next) {
        Node* tmp = curr->next;
        curr->next = tmp->next;
        delete tmp;
    }
}

void list_clear(Node*& head) {
    while (head) {
        Node* tmp = head;
        head = head->next;
        delete tmp;
    }
}

// 1.b) стек

struct Stack {
    Node* top = nullptr;
};

void push(Stack& s, int value) {
    list_add(s.top, value);
}

int pop(Stack& s) {
    int val = s.top->data;
    Node* tmp = s.top;
    s.top = s.top->next;
    delete tmp;
    return val;
}

bool isEmpty(Stack& s) {
    return s.top == nullptr;
}

// 1.c) очередь

struct Queue {
    Node* front = nullptr;
    Node* back = nullptr;
};

void enqueue(Queue& q, int value) {
    Node* newNode = new Node{value, nullptr};
    if (!q.back) {
        q.front = q.back = newNode;
    } else {
        q.back->next = newNode;
        q.back = newNode;
    }
}

int dequeue(Queue& q) {
    int val = q.front->data;
    Node* tmp = q.front;
    q.front = q.front->next;
    if (!q.front) q.back = nullptr;
    delete tmp;
    return val;
}

bool isEmpty(Queue& q) {
    return q.front == nullptr;
}

// 2) параллельная работа со структурами данных

void part2_structures() {
    cout << "ЧАСТЬ 2. СТРУКТУРЫ ДАННЫХ\n";

    Node* list = nullptr;

    // 2.a) параллельное добавление элементов в список
    auto start = high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < 1000; i++) {
        #pragma omp critical
        list_add(list, i);
    }

    auto end = high_resolution_clock::now();

    cout << "поиск элемента 500: "
         << (list_find(list, 500) ? "найден" : "не найден") << endl;

    // 2.b) измерение времени
    cout << "время добавления: "
         << duration_cast<milliseconds>(end - start).count()
         << " мс\n";

    Stack s;
    push(s, 10);
    push(s, 20);
    cout << "pop из стека: " << pop(s) << endl;

    Queue q;
    enqueue(q, 1);
    enqueue(q, 2);
    cout << "dequeue из очереди: " << dequeue(q) << endl;

    list_clear(list);
    cout << endl;
}

/* ЧАСТЬ 3. ДИНАМИЧЕСКАЯ ПАМЯТЬ И УКАЗАТЕЛИ */

// 2) функция вычисления среднего значения
double average_parallel(int* arr, int N) {
    long long sum = 0;

    // 3.a) параллельный подсчёт суммы
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }

    return (double)sum / N;
}

void part3_dynamic_memory() {
    cout << "ЧАСТЬ 3. ДИНАМИЧЕСКАЯ ПАМЯТЬ\n";

    // 1) создание динамического массива
    int N = 100000;
    int* arr = new int[N];

    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100 + 1;
    }

    double avg = average_parallel(arr, N);
    cout << "среднее значение массива: " << avg << endl;

    // 4) освобождение памяти
    delete[] arr;
    cout << "память освобождена\n\n";
}

int main() {
    part1_arrays();
    part2_structures();
    part3_dynamic_memory();
    return 0;
}
