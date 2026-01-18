#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CAPACITY 1024
#define THREADS 256

// ==============================
// STACK (LIFO)
// ==============================
struct Stack {
    int* data;
    int* top;
};

__device__ bool stack_push(Stack s, int v) {
    int pos = atomicAdd(s.top, 1);
    if (pos < CAPACITY) {
        s.data[pos] = v;
        return true;
    }
    return false;
}

__device__ bool stack_pop(Stack s, int* v) {
    int pos = atomicSub(s.top, 1) - 1;
    if (pos >= 0) {
        *v = s.data[pos];
        return true;
    }
    return false;
}

__global__ void stack_kernel(Stack s, int* out) {
    int tid = threadIdx.x;
    stack_push(s, tid);
    __syncthreads();
    stack_pop(s, &out[tid]);
}

// ==============================
// QUEUE (FIFO)
// ==============================
struct Queue {
    int* data;
    int* head;
    int* tail;
};

__device__ bool enqueue(Queue q, int v) {
    int pos = atomicAdd(q.tail, 1);
    if (pos < CAPACITY) {
        q.data[pos % CAPACITY] = v;
        return true;
    }
    return false;
}

__device__ bool dequeue(Queue q, int* v) {
    int pos = atomicAdd(q.head, 1);
    if (pos < *q.tail) {
        *v = q.data[pos % CAPACITY];
        return true;
    }
    return false;
}

__global__ void queue_kernel(Queue q, int* out) {
    int tid = threadIdx.x;
    enqueue(q, tid);
    __syncthreads();
    dequeue(q, &out[tid]);
}

// ==============================
// MPMC QUEUE (упрощенная)
// ==============================
struct MPMCQueue {
    int* data;
    int* head;
    int* tail;
};

__device__ void mpmc_enqueue(MPMCQueue q, int v) {
    int pos = atomicAdd(q.tail, 1);
    q.data[pos % CAPACITY] = v;
}

__device__ int mpmc_dequeue(MPMCQueue q) {
    int pos = atomicAdd(q.head, 1);
    return q.data[pos % CAPACITY];
}

__global__ void mpmc_kernel(MPMCQueue q, int* out) {
    int tid = threadIdx.x;
    mpmc_enqueue(q, tid);
    __syncthreads();
    out[tid] = mpmc_dequeue(q);
}

// ==============================
// CPU versions (comparison)
// ==============================
void cpu_stack(int* out) {
    int stack[CAPACITY];
    int top = 0;
    for (int i = 0; i < THREADS; i++) stack[top++] = i;
    for (int i = 0; i < THREADS; i++) out[i] = stack[--top];
}

void cpu_queue(int* out) {
    int q[CAPACITY];
    int h = 0, t = 0;
    for (int i = 0; i < THREADS; i++) q[t++] = i;
    for (int i = 0; i < THREADS; i++) out[i] = q[h++];
}

// ==============================
// MAIN
// ==============================
int main() {
    int *d_data, *d_top, *d_head, *d_tail, *d_out;
    cudaMalloc(&d_data, CAPACITY * sizeof(int));
    cudaMalloc(&d_top, sizeof(int));
    cudaMalloc(&d_head, sizeof(int));
    cudaMalloc(&d_tail, sizeof(int));
    cudaMalloc(&d_out, THREADS * sizeof(int));

    cudaMemset(d_top, 0, sizeof(int));
    cudaMemset(d_head, 0, sizeof(int));
    cudaMemset(d_tail, 0, sizeof(int));

    Stack s{d_data, d_top};
    Queue q{d_data, d_head, d_tail};
    MPMCQueue mq{d_data, d_head, d_tail};

    // ---- STACK ----
    stack_kernel<<<1, THREADS>>>(s, d_out);
    int h_stack[THREADS];
    cudaMemcpy(h_stack, d_out, sizeof(h_stack), cudaMemcpyDeviceToHost);

    std::cout << "Stack (LIFO), first 10:\n";
    for (int i = 0; i < 10; i++) std::cout << h_stack[i] << " ";
    std::cout << "\n";

    // ---- QUEUE ----
    cudaMemset(d_head, 0, sizeof(int));
    cudaMemset(d_tail, 0, sizeof(int));
    queue_kernel<<<1, THREADS>>>(q, d_out);
    int h_queue[THREADS];
    cudaMemcpy(h_queue, d_out, sizeof(h_queue), cudaMemcpyDeviceToHost);

    std::cout << "Queue (FIFO), first 10:\n";
    for (int i = 0; i < 10; i++) std::cout << h_queue[i] << " ";
    std::cout << "\n";

    // ---- MPMC ----
    cudaMemset(d_head, 0, sizeof(int));
    cudaMemset(d_tail, 0, sizeof(int));
    mpmc_kernel<<<1, THREADS>>>(mq, d_out);
    int h_mpmc[THREADS];
    cudaMemcpy(h_mpmc, d_out, sizeof(h_mpmc), cudaMemcpyDeviceToHost);

    std::cout << "MPMC Queue, first 10:\n";
    for (int i = 0; i < 10; i++) std::cout << h_mpmc[i] << " ";
    std::cout << "\n";

    // ---- CPU comparison ----
    int cpu_out[THREADS];
    cpu_stack(cpu_out);
    std::cout << "CPU Stack, first 10:\n";
    for (int i = 0; i < 10; i++) std::cout << cpu_out[i] << " ";
    std::cout << "\n";

    cpu_queue(cpu_out);
    std::cout << "CPU Queue, first 10:\n";
    for (int i = 0; i < 10; i++) std::cout << cpu_out[i] << " ";
    std::cout << "\n";

    return 0;
}
