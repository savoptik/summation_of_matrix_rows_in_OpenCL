//
//  main.cpp
//  summation_of_matrix_rows_in_OpenCL
//
//  Created by Артём Семёнов on 26/09/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//
// время суммирования:
// CPU: 116.751
// GPU:1078.59
// cp opencl: 128.403
//

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <ctime>

const int rows = 100000, cols = 1024; // размеры будущей матрицы

void matrixGeneration(std::vector<float>& vec); // функция генерирует случайную линеризованную матрицу
void sumInCP(std::vector<float>& mat, const int r, const int c, std::vector<float>& res); // функция суммирует строки на процессоре

int main(int argc, char** argv)
{
    std::vector<float> matrix, res1, res2; // вектор матрицы и векторы результатов
    matrix.resize(cols * rows); // выделяем память для матрицы
    res1.resize(rows); // выделяем память для первого массива результатов.
    res2.resize(rows); // выделяем память для второго массива результатов
    matrixGeneration(matrix); // генирируем матрицу случайных чисел
    sumInCP(matrix, rows, cols, res1); // выполняем суммирование на процессоре.
    
    int err; // ошибка
    char *KernelSource = (char*) malloc(1000000); // указатель на буфер со строкой - кодом kernel-функции
    
    size_t global; // размер грида ()в потоках)
    size_t local; // размер блока потоков
    
    cl_device_id device_id; // id вычислителя
    cl_context context;  // контекст
    cl_command_queue commands; // очередь заданий для выполнения на видеокарте
    cl_program program; // наша программа для видеокарты (она сейчас состоит всего из одной функции)
    cl_kernel kernel; // объект, соответствующий нашей kernel-функции
    
    cl_mem input; // буфер для входных данных на видеокарте
    cl_mem output; // буфер для выходных данных на видеокарте
    
    // ищем вычислительное устройство нужного типа
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
      char device_string[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf("  CL_DEVICE_NAME: \t\t\t%s\n", device_string);
    
    // создаем контекст
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // создаем очередь задач
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // читаем код kernel-функции из файла
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("Error: Не могу прочесть файл с кодом kernel-функции!\n");
        return EXIT_FAILURE;
    }
    char symbol;
    int i=0;
    while((symbol = getc(fp)) != EOF) {
        KernelSource[i++] = symbol;
    }
    fclose(fp);
    KernelSource[i] = 0;
    //    printf("%s", KernelSource);
    
    // создаем объект-программу для видеокарты
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    // компилируем программу для видеокарты
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    // находим в программе для видеокарты точку входа - kernel функию с нужным именем
    kernel = clCreateKernel(program, "sumInRow", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    
    // выделяем память на видеокарте для входных данных и результата
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * matrix.size(), NULL, NULL);
    if (!input)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    output = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * rows, NULL, NULL);
    if (!output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // копируем входные данные на видеокарту
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * matrix.size(), matrix.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    // задаем параметры вызова kernel-функции
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &cols);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &rows);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    // определяем максимальные размер блока потоков
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    
    // запускаем нашу kernel-функцию на гриде из count потоков с найденным максимальным размером блока
    global = (matrix.size() + local - 1) / local * local;
    cl_event event;
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &event);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    
// ждем завершения выполнения задачи
    err = clWaitForEvents(1, &event);
//    clFinish(commands);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    cl_ulong time_start, time_end;
    // получаем время начала и конца вычисления
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    if (err != CL_SUCCESS) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "Время crono: " << (double)duration / 1000 << " ошибка номер " << err << " а должно быть " << CL_SUCCESS << std::endl;
    } else std::cout << "время opencl: " << (double)(time_end - time_start)/1e6 << " ошибка номер " << err << std::endl;
    clReleaseEvent(event);
    
    // копируем результаты с видеокарты
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * res2.size(), res2.data(), 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // сверяемся
    uint good = 0;
    for (uint i = 0; i < res2.size(); i++) {
        good = res2[i] == res1[i] ? good+1: good;
    }
    if (good == rows) {
        std::cout << "Всё хорошо\n";
    } else std::cout << "Что-то пошло не так\n";
    printf("я всё.\n");
    
    // освобождаем память
    clReleaseMemObject(input);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    return 0;
}

void matrixGeneration(std::vector<float>& vec) {
    std::mt19937 gen(static_cast<int>(time(0))); // создаём генератор.
    std::uniform_real_distribution<float> urd(-1.0, 1.0); // задаём диапазон
    for (long i = 0; i < vec.size(); i++) { // идём по массиву
        vec[i] = urd(gen); // заполняем его случайными числами
    }
}

void sumInCP(std::vector<float>& mat, const int r, const int c, std::vector<float>& res) {
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now(); // фиксируем время начала вычияления
    for (int i = 0; i < r; i++) {
        float sum = 0;
        int cr = i * c;
        for (int j = 0; j < c; j++) {
            sum += mat[cr + j];
        }
        res[i] = sum;
    }
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); // фиксируем время конца вычисления
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(); // получаем время выполнения в микросекундах
    std::cout << "Время на ЦП: " << (double)duration / 1000 << std::endl; // выводим время в милисекундах
}
