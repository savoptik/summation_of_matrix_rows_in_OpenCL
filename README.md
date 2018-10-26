# Суммирование строк матрицы
## Задание:
+ Выполнить суммирование строк матрицы значений float 32 размером 960 столбцов на 100000 строк;
+ Замерить скорость вычисления на процессоре и на видеокарте;
+ Оптимезировать программу видеокарты для получения ускорения;
+ Вычислить теоретическое время выполнения и сравнить его с полученным.

Задача выполнялась на видеокарте HD Graphics 4000

## Полученные результаты:
+ На процессоре: 104.422 мсек;
+ На видеокарте: без оптимизации: 608.091 мсек против теоретических 1467 мсек;
+ На видеокарте с оптимизацией: 118.982 мсек против теоретических 98.512.

Так же осуществлена попытка запустить программу на процессоре Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz со следующим результатом:
+ Не оптимизированнное ядро: 59.1073 мсек;
+ Оптимизированная версия не запустилась.
