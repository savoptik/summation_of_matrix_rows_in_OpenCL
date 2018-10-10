__kernel void sumInRow(__global float* input, unsigned int size_row, __global float* output, unsigned int size_output)
{
    unsigned int i = get_global_id(0); // получаем индекс потока
    if (i >= size_output) return; // выходим, если идентификатор потоков больше длинны выходного массива.
    unsigned int currentRow = i * size_row; // вычисляем номер текущей строки.
    float sum = 0; // переменная для накопления суммы.
    for (uint j = 0; j < size_row; j++) {
        sum += input[currentRow + j]; // суммируем строку
    }
    output[i] = sum; // переписываем значение в выходной массив.
}

__kernel void sumInRowFast(__global float* input, unsigned int size_row, __global float* output, unsigned int size_output)
{
    unsigned int i = get_global_id(0); // получаем индекс потока
    if (i >= size_output) return;
    unsigned int th = get_local_id(0); // получаем индекс потока
    __local float cache[128][3*32]; // заводим свой управляемый кеш
    float sum = 0; // переменная для накопления суммы.
    for (int part = 0; part < 10; part++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int r = 0; r < 96; r++) cache[th+r/3][th%96] = input[i+r/3+th%96];
        for (int c = 0; c < 3*32; c++) sum += cache[th][c];
    }
    output[i] = sum; // переписываем значение в выходной массив.
}
