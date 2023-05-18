import csv
import pandas
import numpy as np

# открываем файл с помощью pandas
file = pandas.read_excel("file.xlsx")
print(file)

# выводим файл в csv без индексов строк
file.to_csv("123.csv", index=False)

# считываем с помощью csv reader
with open("123.csv", 'r') as pre_table:
    table = csv.reader(pre_table)
    A_list = [list(map(int, row)) for row in table]

# делаем матрицу numpy
A = np.array(A_list)
print(A)
# транспонированная матрица B
B = A.T
print(B)
# сумма элементов у произведения матриц
C = np.dot(A, B)
sum_C = np.sum(C)
print("Сумма элементов матрицы C = AB:", sum_C)
# находим сумму элементов по каждому столбцу матрицы A
d = np.sum(A, axis=0)
sum_d_squares = np.sum(np.square(d))
print("Сумма квадратов элементов d:", sum_d_squares)

# Сумма элементов матрицы C = AB: 174384
# Сумма квадратов элементов d: 174384
