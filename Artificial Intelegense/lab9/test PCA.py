import numpy as np
import matplotlib.pyplot as plt

# создаем данные для построения контура
x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = np.sin(X-37) * np.cos(Y+30)

# создаем контурную диаграмму
plt.contourf(X, Y, Z, cmap='plasma')

# добавляем легенду и название графика
plt.colorbar()
plt.title('Contour Plot')

# отображаем график
plt.show()
