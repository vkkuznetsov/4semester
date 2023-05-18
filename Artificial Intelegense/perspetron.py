num0 = list('1100')  # верная картинка наклонной линии
num1 = list('1111')  # другие картинки
num2 = list('0110')
num3 = list('0011')


# Функция персептрона -возвращает 1
# если набор сигналов соответсвует шаблону, задаваемому весами.
def perceptron(Sensor, Weights, Bias):
    s = 0  # Начальное значение суммы
    n_sensor = len(Sensor)  # количество сенсоров
    for i in range(n_sensor):  # цикл суммирования сигналов от сенсоров
        s += int(Sensor[i]) * Weights[i]
    if s >= Bias:
        return 1  # Сумма превысила порог
    else:
        return 0  # Сумма меньше порога


def IsincFullSquare(Sensor):
    weights_i1 = [1, 1, 1, 1]
    res = perceptron(Sensor, weights_i1, 4)
    return res == 1


def IsDot(Sensor):
    weights_i1 = [1, -1, -1, -1]
    weights_i2 = [-1, 1, -1, -1]
    weights_i3 = [-1, -1, 1, -1]
    weights_i4 = [-1, -1, -1, 1]
    a1 = perceptron(Sensor, weights_i1, 1)
    a2 = perceptron(Sensor, weights_i2, 1)
    a3 = perceptron(Sensor, weights_i3, 1)
    a4 = perceptron(Sensor, weights_i4, 1)
    res = perceptron([a1, a2, a3, a4], [1, 1, 1, 1], 1)
    return res == 1


def IsHorizontal(Sensor):
    weights_i1 = [1, 1, -1, -1]
    weights_i2 = [-1, -1, 1, 1]
    a1 = perceptron(Sensor, weights_i1, 2)
    a2 = perceptron(Sensor, weights_i2, 2)
    res = perceptron([a1, a2], [1, 1], 1)
    return res == 1


def IsVertical(Sensor):
    weights_i1 = [1, -1, 1, -1]
    weights_i2 = [-1, 1, -1, 1]
    a1 = perceptron(Sensor, weights_i1, 2)
    a2 = perceptron(Sensor, weights_i2, 2)
    res = perceptron([a1, a2], [1, 1], 1)
    return res == 1


def IsInclinedLine(Sensor):
    weights_i1 = [1, -1, -1, 1]
    weights_i2 = [-1, 1, 1, -1]
    a1 = perceptron(Sensor, weights_i1, 2)
    a2 = perceptron(Sensor, weights_i2, 2)
    res = perceptron([a1, a2], [1, 1], 1)
    return res == 1



# передаю картинку, проверяю по всем функциям ее и смотрю под какие подходит, если выдает True то говорю называние функции на которой сработал
def KnowWhatIs(res):
    print("Определено автоматом через функцию")
    if IsDot(res):
        print(f"{res}--Это точка")
    elif IsHorizontal(res):
        print(f"{res}--это горизонтальная")
    elif IsVertical(res):
        print(f"{res}--Это вертикальная")
    elif IsincFullSquare(res):
        print(f"{res}--Это полный квадрат")
    elif IsInclinedLine(res):
        print(f"{res}--Это наклонная линия")
    else:
        print("это что-то другое!")


print(num0, "это наклонная? ", IsInclinedLine(num0))
print(num1, "это наклонная? ", IsInclinedLine(num1))
print(num2, "это наклонная? ", IsInclinedLine(num2))
print(num3, "это наклонная? ", IsInclinedLine(num3))

print(num0, "это квадрат?", IsincFullSquare(num0))
print(num1, "это квадрат?", IsincFullSquare(num1))
print(num2, "это квадрат?", IsincFullSquare(num2))
print(num3, "это квадрат?", IsincFullSquare(num3))

print(num0, "это точка?", IsDot(num0))
print(num1, "это точка?", IsDot(num1))
print(num2, "это точка?", IsDot(num2))
print(num3, "это точка?", IsDot(num3))

print(num0, "это горизонтальтная", IsHorizontal(num0))
print(num1, "это горизонтальтная", IsHorizontal(num1))
print(num2, "это горизонтальтная", IsHorizontal(num2))
print(num3, "это горизонтальтная", IsHorizontal(num3))

print(num0, "это вертикальная", IsVertical(num0))
print(num1, "это вертикальная", IsVertical(num1))
print(num2, "это вертикальная", IsVertical(num2))
print(num3, "это вертикальная", IsVertical(num3))

KnowWhatIs(num0)
KnowWhatIs(num1)
KnowWhatIs(num2)
KnowWhatIs(num3)

# #['1', '1', '0', '0'] это наклонная?  False
# ['1', '1', '1', '1'] это наклонная?  False
# ['0', '1', '1', '0'] это наклонная?  True
# ['0', '0', '1', '1'] это наклонная?  False
# ['1', '1', '0', '0'] это квадрат? False
# ['1', '1', '1', '1'] это квадрат? True
# ['0', '1', '1', '0'] это квадрат? False
# ['0', '0', '1', '1'] это квадрат? False
# ['1', '1', '0', '0'] это точка? False
# ['1', '1', '1', '1'] это точка? False
# ['0', '1', '1', '0'] это точка? False
# ['0', '0', '1', '1'] это точка? False
# ['1', '1', '0', '0'] это горизонтальтная True
# ['1', '1', '1', '1'] это горизонтальтная False
# ['0', '1', '1', '0'] это горизонтальтная False
# ['0', '0', '1', '1'] это горизонтальтная True
# ['1', '1', '0', '0'] это вертикальная False
# ['1', '1', '1', '1'] это вертикальная False
# ['0', '1', '1', '0'] это вертикальная False
# ['0', '0', '1', '1'] это вертикальная False
# Определено автоматом через функцию
# ['1', '1', '0', '0']--это горизонтальная
# Определено автоматом через функцию
# ['1', '1', '1', '1']--Это полный квадрат
# Определено автоматом через функцию
# ['0', '1', '1', '0']--Это наклонная линия
# Определено автоматом через функцию
# ['0', '0', '1', '1']--это горизонтальная