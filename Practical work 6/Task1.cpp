#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

int main() {

    // Размер входных массивов
    const int N = 1000000;

    // Инициализация данных
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N, 0.0f);

    // Получение доступных OpenCL платформ
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    std::cout << "OpenCL platforms found: " << platformCount << std::endl;

    // Перебор всех платформ
    for (cl_uint p = 0; p < platformCount; ++p) {
        cl_platform_id platform = platforms[p];

        // Получение устройств платформы
        cl_uint deviceCount = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

        // Перебор устройств
        for (cl_uint d = 0; d < deviceCount; ++d) {
            cl_device_id device = devices[d];

            // Определение типа устройства
            cl_device_type deviceType;
            clGetDeviceInfo(device, CL_DEVICE_TYPE,
                            sizeof(cl_device_type), &deviceType, nullptr);

            std::string typeName =
                (deviceType == CL_DEVICE_TYPE_CPU) ? "CPU" :
                (deviceType == CL_DEVICE_TYPE_GPU) ? "GPU" : "Other";

            std::cout << "\nPlatform " << p
                      << ", Device " << d
                      << " (" << typeName << ")" << std::endl;

            // Создание контекста
            cl_int err;
            cl_context context =
                clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) continue;

            // Создание очереди команд
            cl_command_queue queue =
                clCreateCommandQueueWithProperties(context, device, nullptr, &err);
            if (err != CL_SUCCESS) {
                clReleaseContext(context);
                continue;
            }

            // Загрузка OpenCL-ядра из файла
            std::ifstream kernelFile("kernel.cl");
            if (!kernelFile.is_open()) {
                std::cout << "Failed to open kernel.cl" << std::endl;
                return 1;
            }

            std::ostringstream kernelStream;
            kernelStream << kernelFile.rdbuf();
            std::string kernelSource = kernelStream.str();
            const char* sourceStr = kernelSource.c_str();

            // Создание и компиляция программы
            cl_program program =
                clCreateProgramWithSource(context, 1, &sourceStr, nullptr, &err);
            err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

            // Вывод лога компиляции при ошибке
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

            // Создание ядра
            cl_kernel kernel =
                clCreateKernel(program, "vector_add", &err);

            // Создание буферов
            cl_mem bufferA =
                clCreateBuffer(context,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * N,
                               A.data(), &err);

            cl_mem bufferB =
                clCreateBuffer(context,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * N,
                               B.data(), &err);

            cl_mem bufferC =
                clCreateBuffer(context,
                               CL_MEM_WRITE_ONLY,
                               sizeof(float) * N,
                               nullptr, &err);

            // Установка аргументов ядра
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

            // Запуск ядра и замер времени
            size_t globalSize = N;
            auto start = std::chrono::high_resolution_clock::now();

            clEnqueueNDRangeKernel(queue, kernel,
                                   1, nullptr, &globalSize,
                                   nullptr, 0, nullptr, nullptr);
            clFinish(queue);

            auto end = std::chrono::high_resolution_clock::now();

            // Чтение результата
            clEnqueueReadBuffer(queue, bufferC, CL_TRUE,
                                0, sizeof(float) * N,
                                C.data(), 0, nullptr, nullptr);

            // Вывод времени и проверки результата
            std::chrono::duration<double, std::milli> elapsed = end - start;
            std::cout << "Execution time: "
                      << elapsed.count() << " ms" << std::endl;

            std::cout << "C[0..4] = ";
            for (int i = 0; i < 5; ++i)
                std::cout << C[i] << " ";
            std::cout << std::endl;

            // Освобождение ресурсов
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