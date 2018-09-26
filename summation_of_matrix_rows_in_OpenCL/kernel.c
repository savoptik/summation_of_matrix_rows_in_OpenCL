typedef unsigned int  uint;

__kernel void sumInRow(__global doubl* input, uint size_row, __global double* autput, uint size_autput)
{
    uint i = get_global_id(0); // получаем индекс потока
    if (i >= size_output) return; // выходим, если идентификатор потоков больше длинны выходного массива.
    uint currentRow = i * size_row; // вычисляем номер текущей строки.
    double sum = 0.0000; // переменная для накопления суммы.
    for (uint j = 0; j < size_row; j++) {
        sum += input[currentRow + j]; // суммируем строки.
    }
    autput[i] = sum; // переписываем значение в выходной массив.
}
