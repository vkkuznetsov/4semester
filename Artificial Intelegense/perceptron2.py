import itertools

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


def perceptron(Sensor, Weights, Bias):
    s = 0  # Начальное значение суммы
    n_sensor = len(Sensor)  # количество сенсоров
    for i in range(n_sensor):  # цикл суммирования сигналов от сенсоров
        s += int(Sensor[i]) * int(Weights[i])
    if s >= Bias:
        return 1  # Сумма превысила порог
    else:
        return 0  # Сумма меньше порога


def isOne(Sensor):
    wight = [-1, 1, 1, -1, -1, -1, 1, -1, -1]
    res = perceptron(Sensor, wight, 3)
    return res == 1


def isZero(Sensor):
    wight = [1, 1, 1, 1, 1, 1, -1, -1, -1]
    res = perceptron(Sensor, wight, 6)
    return res == 1


def isTwo(Sensor):
    weight = [1, 1, -1, 1, -1, -1, -1, -1, 1]
    res = perceptron(Sensor, weight, 4)
    return res == 1


def isThree(Sensor):
    weight = [1, -1, -1, -1, -1, -1, 1, 1, 1]
    res = perceptron(Sensor, weight, 4)
    return res == 1


def isFour(Sensor):
    weight = [-1, 1, 1, -1, -1, 1, -1, 1, -1]
    res = perceptron(Sensor, weight, 4)
    return res == 1


def isFive(Sensor):
    weight = [1, -1, 1, 1, -1, 1, -1, 1, -1]
    res = perceptron(Sensor, weight, 5)
    return res == 1


def isSix(Sensor):
    weight = [-1, -1, 1, 1, 1, -1, 1, 1, -1]
    res = perceptron(Sensor, weight, 5)
    return res == 1


def isSeven(Sensor):
    weight = [1, -1, -1, -1, 1, -1, 1, -1, -1]
    res = perceptron(Sensor, weight, 3)

    return res == 1


def isEight(Sensor):
    wight = [1, 1, 1, 1, 1, 1, -1, 1, -1]
    res = perceptron(Sensor, wight, 7)
    return res == 1


def isNine(Sensor):
    weight = [1, 1, -1, -1, -1, 1, -1, 1, 1]
    res = perceptron(Sensor, weight, 5)
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


for i in nums:
    print(know_what_is_the_number(i))

combinations = list(itertools.product([0, 1], repeat=9))  # создаем все возможные списки
nums = [[int(num) for num in sublist] for sublist in nums]  # надо чтобы можно было сравнивать
count = 0
allcount = 0
for i in combinations:

    if list(i) not in nums:
        allcount += 1
        print(i, end=" ")
        print(know_what_is_the_number(i))
        if know_what_is_the_number(i) == "не знаю":
            count += 1
print(count, allcount, count / allcount * 100, "%")
