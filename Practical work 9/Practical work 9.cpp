#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

/* =========================================================
   TASK 1 — Mean and Standard Deviation
   ========================================================= */
void task1(int rank, int size) {
    const int N = 1000000;
    vector<double> data;

    vector<int> counts(size), displs(size);

    if (rank == 0) {
        data.resize(N);
        for (int i = 0; i < N; i++)
            data[i] = rand() % 100;
    }

    int base = N / size;
    int rem = N % size;

    for (int i = 0; i < size; i++) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
    }

    vector<double> local(counts[rank]);

    MPI_Scatterv(data.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sum = 0, local_sq = 0;
    for (double x : local) {
        local_sum += x;
        local_sq += x * x;
    }

    double sum = 0, sq = 0;
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq, &sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double mean = sum / N;
        double stddev = sqrt(sq / N - mean * mean);
        cout << "\nTASK 1\nMean = " << mean << "\nStdDev = " << stddev << endl;
    }
}

/* =========================================================
   TASK 2 — Gaussian Elimination
   ========================================================= */
void task2(int rank, int size) {
    const int N = 4;
    vector<double> A(N*N), b(N), x(N);

    if (rank == 0) {
        double A0[4][4] = {
            {10,2,1,1},
            {2,10,1,1},
            {1,1,10,2},
            {1,1,2,10}
        };
        double b0[4] = {14,14,14,14};

        for (int i = 0; i < N; i++) {
            b[i] = b0[i];
            for (int j = 0; j < N; j++)
                A[i*N + j] = A0[i][j];
        }
    }

    for (int k = 0; k < N; k++) {
        MPI_Bcast(&A[k*N], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = k + 1 + rank; i < N; i += size) {
            double f = A[i*N + k] / A[k*N + k];
            for (int j = k; j < N; j++)
                A[i*N + j] -= f * A[k*N + j];
            b[i] -= f * b[k];
        }
    }

    if (rank == 0) {
        for (int i = N - 1; i >= 0; i--) {
            x[i] = b[i];
            for (int j = i + 1; j < N; j++)
                x[i] -= A[i*N + j] * x[j];
            x[i] /= A[i*N + i];
        }

        cout << "\nTASK 2 Solution:\n";
        for (int i = 0; i < N; i++)
            cout << "x" << i << " = " << x[i] << endl;
    }
}

/* =========================================================
   TASK 3 — Floyd–Warshall
   ========================================================= */
void task3(int rank, int size) {
    const int N = 4;
    vector<double> G;

    if (rank == 0) {
        G = {
            0, 5, 1, 999,
            5, 0, 2, 3,
            1, 2, 0, 4,
            999, 3, 4, 0
        };
    }

    int rows = N / size;
    vector<double> local(rows * N);

    MPI_Scatter(G.data(), rows*N, MPI_DOUBLE, local.data(), rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> full(N*N);

    for (int k = 0; k < N; k++) {
        MPI_Allgather(local.data(), rows*N, MPI_DOUBLE, full.data(), rows*N, MPI_DOUBLE, MPI_COMM_WORLD);

        for (int i = 0; i < rows; i++) {
            int gi = rank * rows + i;
            for (int j = 0; j < N; j++) {
                double d = full[gi*N + k] + full[k*N + j];
                if (d < local[i*N + j])
                    local[i*N + j] = d;
            }
        }
    }

    MPI_Gather(local.data(), rows*N, MPI_DOUBLE, G.data(), rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "\nTASK 3 Result:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                cout << G[i*N + j] << " ";
            cout << endl;
        }
    }
}

/* ========================================================= */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t1 = MPI_Wtime();
    task1(rank, size);
    task2(rank, size);
    task3(rank, size);
    double t2 = MPI_Wtime();

    if (rank == 0)
        cout << "\nTotal execution time: " << t2 - t1 << " seconds\n";

    MPI_Finalize();
    return 0;
}
