#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <clocale>

int main() {

    std::setlocale(LC_ALL, "Russian");

    // Размеры матриц
    const int N = 4;   // строки A
    const int M = 3;   // столбцы A / строки B
    const int K = 5;   // столбцы B

    // Матрицы
    std::vector<float> A(N * M);
    std::vector<float> B(M * K);
    std::vector<float> C(N * K, 0.0f);

    // Инициализация данных
    for (int i = 0; i < N * M; ++i) A[i] = i + 1;
    for (int i = 0; i < M * K; ++i) B[i] = 1.0f;

    // Последовательная версия (CPU)
    std::vector<float> C_seq(N * K, 0.0f);
    auto start_seq = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * K + j];
            C_seq[i * K + j] = sum;
        }

    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = end_seq - start_seq;
    std::cout << "CPU time: " << cpuTime.count() << " ms\n";

    // Получение платформ OpenCL
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    // Перебор платформ и устройств
    for (cl_uint p = 0; p < platformCount; ++p) {
        cl_platform_id platform = platforms[p];

        cl_uint deviceCount = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

        for (cl_uint d = 0; d < deviceCount; ++d) {
            cl_device_id device = devices[d];

            cl_int err;
            cl_context context =
                clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) continue;

            cl_command_queue queue =
                clCreateCommandQueueWithProperties(context, device, nullptr, &err);
            if (err != CL_SUCCESS) {
                clReleaseContext(context);
                continue;
            }

            // Загрузка ядра
            std::ifstream kernelFile("matrix_mul.cl");
            std::ostringstream kernelStream;
            kernelStream << kernelFile.rdbuf();
            std::string kernelSource = kernelStream.str();
            const char* srcStr = kernelSource.c_str();

            cl_program program =
                clCreateProgramWithSource(context, 1, &srcStr, nullptr, &err);
            err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

            if (err != CL_SUCCESS) {
                size_t logSize;
                clGetProgramBuildInfo(program, device,
                                      CL_PROGRAM_BUILD_LOG,
                                      0, nullptr, &logSize);
                std::vector<char> buildLog(logSize);
                clGetProgramBuildInfo(program, device,
                                      CL_PROGRAM_BUILD_LOG,
                                      logSize, buildLog.data(), nullptr);
                std::cout << buildLog.data() << std::endl;
            }

            cl_kernel kernel =
                clCreateKernel(program, "matrix_mul", &err);

            // Буферы
            cl_mem bufferA =
                clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * A.size(), A.data(), &err);

            cl_mem bufferB =
                clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * B.size(), B.data(), &err);

            cl_mem bufferC =
                clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                               sizeof(float) * C.size(), nullptr, &err);

            // Аргументы ядра
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
            clSetKernelArg(kernel, 3, sizeof(int), &N);
            clSetKernelArg(kernel, 4, sizeof(int), &M);
            clSetKernelArg(kernel, 5, sizeof(int), &K);

            size_t globalSize[2] = { (size_t)N, (size_t)K };

            // Запуск OpenCL-версии
            auto start = std::chrono::high_resolution_clock::now();
            clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                   globalSize, nullptr, 0, nullptr, nullptr);
            clFinish(queue);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> gpuTime = end - start;

            clEnqueueReadBuffer(queue, bufferC, CL_TRUE,
                                0, sizeof(float) * C.size(),
                                C.data(), 0, nullptr, nullptr);

            // Вывод результата
            std::cout << "Platform " << p
                      << ", Device " << d
                      << " time: " << gpuTime.count() << " ms\n";

            std::cout << "C =\n";
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < K; ++j)
                    std::cout << C[i * K + j] << " ";
                std::cout << "\n";
            }

            // Проверка корректности
            bool correct = true;
            for (int i = 0; i < N * K; ++i)
                if (std::fabs(C[i] - C_seq[i]) > 1e-5)
                    correct = false;

            std::cout << (correct ? "Результат верный\n\n"
                                   : "Ошибка в вычислениях\n");

            // Очистка ресурсов
            clReleaseMemObject(bufferA);
            clReleaseMemObject(bufferB);
            clReleaseMemObject(bufferC);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
    }

    return 0;
}