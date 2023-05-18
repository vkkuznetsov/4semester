import random
import itertools
import matplotlib.pyplot as plt
import numpy as np


num0 = list('111111000')
num1 = list('011000100')
num2 = list('110100001')
num3 = list('100000111')
num4 = list('011001010')
num5 = list('101101010')
num6 = list('001110110')
num7 = list('100010100')
num8 = list('111111010')
num9 = list('110001011')
nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]

n_sensor = 9
weights = [0 for i in range(n_sensor)]
print(weights)


def perceptron(Sensor):
    b = 5
    s = 0
    for i in range(n_sensor):
        s += int(Sensor[i]) * int(weights[i])
    if s >= b:
        return True
    return False


def perceptron2(Sensor, weights):
    b = 5
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


super_weights = [[], [], [], [], [], [], [], [], [], []]
EPOCH = 20000
for dig in range(0, 10):
    weights = [0 for i in range(9)]
    print(f"Обучаем число {dig}")
    for i in range(EPOCH):
        j = random.randint(0, 9)
        r = perceptron(nums[j])
        if j != dig:
            if r:
                decrease(nums[j])
        else:
            if not r:
                increase(nums[j])

        temp_weight = tuple(weights)
        super_weights[dig] = temp_weight
        if i + 1 in [1, 10, 100, 1000, 5000, 50000, 100000, 200000]:
            print(f" {i + 1}\t{weights},{perceptron(nums[dig])}")

    print("Проверка обучения")
    for i in range(10):
        print(i, f"это {dig}?", perceptron(nums[i]))


# for k in range(10):
#     for i in range(10):
#         print(k, end="= ")
#         print(i, perceptron2(nums[i], super_weights[k]))


def isZero(Sensor):
    res = perceptron2(Sensor, super_weights[0])
    return res == 1


def isOne(Sensor):
    res = perceptron2(Sensor, super_weights[1])
    return res == 1


def isTwo(Sensor):
    res = perceptron2(Sensor, super_weights[2])
    return res == 1


def isThree(Sensor):
    res = perceptron2(Sensor, super_weights[3])
    return res == 1


def isFour(Sensor):
    res = perceptron2(Sensor, super_weights[4])
    return res == 1


def isFive(Sensor):
    res = perceptron2(Sensor, super_weights[5])
    return res == 1


def isSix(Sensor):
    res = perceptron2(Sensor, super_weights[6])
    return res == 1


def isSeven(Sensor):
    res = perceptron2(Sensor, super_weights[7])
    return res == 1


def isEight(Sensor):
    res = perceptron2(Sensor, super_weights[8])
    return res == 1


def isNine(Sensor):
    res = perceptron2(Sensor, super_weights[9])
    return res == 1


def know_what_is_the_number(Sensor):
    if isOne(Sensor):
        return "Это число один"
    elif isZero(Sensor):
        return "Это число ноль"
    elif isTwo(Sensor):
        return "Это число два"
    elif isThree(Sensor):
        return "Это число три"
    elif isFour(Sensor):
        return "Это число четыре"
    elif isFive(Sensor):
        return "Это число пять"
    elif isSix(Sensor):
        return "Это число шесть"
    elif isSeven(Sensor):
        return "Это число семь"
    elif isEight(Sensor):
        return "Это число восемь"
    elif isNine(Sensor):
        return "Это число девять"
    else:
        return "не знаю"

def know_what_is_the_number2(Sensor):
    if isOne(Sensor):
        return 1
    elif isZero(Sensor):
        return 0
    elif isTwo(Sensor):
        return 2
    elif isThree(Sensor):
        return 3
    elif isFour(Sensor):
        return 4
    elif isFive(Sensor):
        return 5
    elif isSix(Sensor):
        return 6
    elif isSeven(Sensor):
        return 7
    elif isEight(Sensor):
        return 8
    elif isNine(Sensor):
        return 9
    else:
        return -1

for i in nums:
    print(know_what_is_the_number(i))

# првоерка работы на некорректных вариантах

combinations = list(itertools.product([0, 1], repeat=9))  # создаем все возможные списки
nums = [[int(num) for num in sublist] for sublist in nums]  # надо чтобы можно было сравнивать
count = 0
all_count = 0

for i in combinations:
    if list(i) not in nums:

        all_count += 1
        print(i, end=" ")
        print(know_what_is_the_number(i))
        if know_what_is_the_number(i) == "не знаю":
            count += 1
print(count, all_count, count / all_count * 100, "%")

# ВИЗУАЛИЗАЦИЯ

# Задаем значения x вручную
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

y = [know_what_is_the_number2(nums[i]) for i in range(10)]

# Создаем график и рисуем линию
fig, ax = plt.subplots()
ax.scatter(x, y)

# Задаем метки на осях
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))

# Выводим график


plt.show()
