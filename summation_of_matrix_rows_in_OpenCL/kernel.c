typedef unsigned int  uint; // мне не хватает в сях этого типа.

__kernel void sumInRow(__global float* input, uint size_row, __global float* autput, uint size_autput)
{
    uint i = get_global_id(0); // получаем индекс потока
    if (i >= size_output) return; // выходим, если идентификатор потоков больше длинны выходного массива.
    uint currentRow = i * size_row; // вычисляем номер текущей строки.
    float sum = 0.0000; // переменная для накопления суммы.
    for (uint j = 0; j < size_row; j++) {
        sum += input[currentRow + j]; // суммируем строку
    }
    autput[i] = sum; // переписываем значение в выходной массив.
}
