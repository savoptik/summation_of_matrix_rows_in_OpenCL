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
