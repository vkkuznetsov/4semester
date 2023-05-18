import random

import matplotlib.pyplot as plt
import numpy as np

result_without_changing = 0
result_with_changing = 0

random.seed(5)

pairs = {}


def create_pairs(n):
    mu = 100  # среднее значение для базовой величины
    sigma1 = 5  # стандартное отклонение для базовой величины
    sigma2 = 10  # стандартное отклонение для роста

    for i in range(n):
        base = random.gauss(mu, sigma1)
        foot = base / 3.6 + random.gauss(0, 0.5)
        height = base * 1.7 + random.gauss(0, sigma2)
        pairs[foot] = height


# создаем 100 пар
create_pairs(100)

# Вычисляем с помощью функций отклонение (к чему будем стремиться)
sorted_pairs = sorted(pairs.items())

# Создаем массивы для x и y
x = np.array([pair[0] for pair in sorted_pairs])
y = np.array([pair[1] for pair in sorted_pairs])

# Вычисляем коэффициенты линейной регрессии (a и b) с помощью МНК
a, b = np.polyfit(x, y, 1)

# Вычисляем среднеквадратичную ошибку (MSE)
y_predicted = a * x + b
mse = np.mean((y_predicted - y) ** 2)

# Выводим коэффициенты линейной регрессии и уравнение прямой

print('УРАВНЕНИЕ ИДЕАЛЬНОЙ ПРЯМОЙ: y =', a, 'x +', b)
print('MSE к КОТОРОМУ НАДО СТРЕМИТЬСЯ =', mse)
plt.plot(x, a * x + b, 'g', label='Какой должна быть')

k = random.uniform(-5, 5)
c = random.uniform(-5, 5)

print(f"Начальная прямая: Y = {k} * X + {c}")


def proceed(x):
    return x * k + c


rate = 0.01
num_step = 1_000_000

errors_dots = []

for i in range(num_step + 1):
    x = random.choice(list(pairs.keys()))
    true_result = pairs[x]
    out = proceed(x)
    delta = true_result - out
    k += delta * rate * (x - 26.5)
    c += delta * rate * 1

    # пересчет ошибки каждую 1000
    if i % 1000 == 0:
        SSE = 0
        for j in range(len(pairs)):
            x = list(pairs.keys())[j]
            true_result = pairs[x]
            out = proceed(x)
            delta = true_result - out
            SSE += delta * delta
        errors_dots.append(SSE / 100 - mse)
        # изменение скорости обучения
        if i % 10_000 == 0:
            rate *= 0.8
        # вывод логов
        if i % 100_000 == 0:
            print("(SSE) = %8f" % (SSE / 100))
            print("Готовая прямая Y = %7f" % k, "X+ %8f" % c)
        if i == 1_000_000:
            result_with_changing = SSE / 100 - mse

z = np.linspace(min(pairs.keys()), max(pairs.keys()), 100)
plt.xlabel('Длина стопы')
plt.ylabel('Рост')
plt.title('С изменением скорости')

plt.scatter(list(pairs.keys()), list(pairs.values()))

plt.plot(z, proceed(z), 'r', label='Какая получилась')

plt.legend()
plt.show()

# -----------------------------------

# Создаем массивы для x и y
x = np.array([pair[0] for pair in sorted_pairs])
y = np.array([pair[1] for pair in sorted_pairs])

# Вычисляем коэффициенты линейной регрессии (a и b) с помощью МНК
a, b = np.polyfit(x, y, 1)

plt.plot(x, a * x + b, 'g', label='Какой должна быть')

k = random.uniform(-5, 5)
c = random.uniform(-5, 5)

print(f"Начальная прямая: Y = {k} * X + {c}")

rate = 0.001
num_step = 1_000_000

errors_dots1 = []

for i in range(num_step + 1):
    x = random.choice(list(pairs.keys()))
    true_result = pairs[x]
    out = proceed(x)
    delta = true_result - out
    k += delta * rate * (x - 26.5)
    c += delta * rate * 1

    # пересчет ошибки каждую 1000
    if i % 1000 == 0:
        SSE = 0
        for j in range(len(pairs)):
            x = list(pairs.keys())[j]
            true_result = pairs[x]
            out = proceed(x)
            delta = true_result - out
            SSE += delta * delta
        errors_dots1.append(SSE / 100 - mse)
        # изменение скорости обучения
        # if i % 10_000 == 0:
        # rate *= 0.9
        # вывод логов
        if i % 100_000 == 0:
            print("(SSE) = %8f" % (SSE / 100))
            print("Готовая прямая Y = %7f" % k, "X+ %8f" % c)
        if i == 1_000_000:
            result_without_changing = SSE / 100 - mse

print("Результаты отклонений")
print(f"Разница когда меняем скорость = {result_with_changing}")
print(f"Разница когда скорость постоянна = {result_without_changing}")

plt.title('Без изменения скорости')
plt.xlabel('Длина стопы')
plt.ylabel('Рост')

plt.scatter(list(pairs.keys()), list(pairs.values()))

plt.plot(z, proceed(z), 'r', label='Какая получилась')
# Отображаем график
plt.legend()
plt.show()

# Вывод ГРАФИКА ОШИБОК
# убираем 1 элемент тк он слишком большой и график не будет видно
errors_dots1.pop(0)
er_x1 = range(len(errors_dots1))
er_y1 = errors_dots1

errors_dots.pop(0)
er_x = range(len(errors_dots))
er_y = errors_dots

plt.plot(er_x1, er_y1, "g", label="Без изменения скорости")
plt.plot(er_x, er_y, "r", label="С изменением скорости")
plt.title("График ошибок")
# Отображаем график
plt.legend()
plt.show()
