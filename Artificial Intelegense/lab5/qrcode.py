import io
import random
import string
import numpy as np
import segno
from matplotlib import pyplot as plt

# max lenght  = 25
random.seed(1)




# переводит из строчного представления в интовые и удаляет знаки новой строки
def str_list_to_int(lst):
    return [int(x) for x in lst if x != '\n']


# случайные строки длиной 25
def randStr(chars=string.ascii_uppercase, N=25):
    return ''.join(random.choice(chars) for _ in range(N))


# Генерируем 100 случайных строк длины 25 из заглавных букв английского алфавита
strings = [''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=25)) for _ in range(100)]

# Генерируем QR-коды для каждой строки
qrcodes = [segno.make(string) for string in strings]
qrcodes_with_mask = [segno.make(string, mask=2) for string in strings]
# 29 строк, каждая весит по 31 и получается 899


# генерирум список списков для макси
qr_matrix_with_mask = []
for i in qrcodes_with_mask:
    buffer = io.StringIO()
    i.save(buffer, kind="txt", border=0)
    result = buffer.getvalue()
    qr_matrix_with_mask.append(str_list_to_int(result))

#  то же самое, но только для того, где нет маски
qr_matrix = []
for i in qrcodes:
    buffer = io.StringIO()
    i.save(buffer, kind="txt", border=0)
    result = buffer.getvalue()
    qr_matrix.append(str_list_to_int(result))


def generate_list_of_lists(num_lists, list_length):
    list_of_lists = []
    for _ in range(num_lists):
        list_of_lists.append([random.choice([0, 1]) for _ in range(list_length)])
    return list_of_lists


# генерируем списки которые не qr
non_qr_matrix = generate_list_of_lists(100, 441)


# функция вывода весов в виде графика
def plot_array(arr):
    # Преобразуем массив в двумерный массив 21x21
    if len(arr) > 441:
        arr = np.array(arr[:441])
    elif len(arr) < 441:
        arr = np.pad(arr, (0, 441 - len(arr)), 'constant', constant_values=0)
    arr = np.array(arr).reshape((21, 21))

    # Отображаем изображение
    plt.imshow(arr)
    plt.show()


# создаем перцептрон
n_sensor = 441


def perceptron(Sensor):
    b = 5000
    s = 0
    for i in range(n_sensor):
        s += int(Sensor[i]) * int(weights[i])
    if s >= b:
        return True
    return False


def decrease(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] -= 1


def increase(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] += 1


# все наши массивы засовываем в 1
allmass = qr_matrix + non_qr_matrix

EPOCH = 20
weights = [0 for i in range(441)]
for i in range(EPOCH):
    random.shuffle(allmass)
    for j in range(200):
        r = perceptron(allmass[j])
        if not allmass[j] in qr_matrix:
            if r:
                decrease(allmass[j])
        else:
            if not r:
                increase(allmass[j])

    # if i + 1 in [1, 10, 15, 20]:
    #     print(f" {i + 1}\t{weights},{perceptron(allmass[j])}")
print("без маски веса = ", len(weights), weights)
plot_array(weights)

allmass = qr_matrix_with_mask + non_qr_matrix

EPOCH = 20
weights = [0 for i in range(441)]
for i in range(EPOCH):
    random.shuffle(allmass)
    for j in range(200):
        r = perceptron(allmass[j])
        if not allmass[j] in qr_matrix_with_mask:
            if r:
                decrease(allmass[j])
        else:
            if not r:
                increase(allmass[j])

    # if i + 1 in [1, 10, 15, 20]:
    #     print(f" {i + 1}\t{weights},{perceptron(allmass[j])}")
print("с маской веса = ", len(weights), weights)
plot_array(weights)
