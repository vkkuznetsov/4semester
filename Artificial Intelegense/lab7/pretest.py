import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


class Perceptron2(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, 0)

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):

                output = self.sigmoid(self.net_input(xi))
                error = target - output
                if target == -1:
                    error = -output
                elif target == 1:
                    error = 1 - output
                self.w_[1:] += self.eta * xi * error * output * (1 - output)
                self.w_[0] += self.eta * error * output * (1 - output)
                errors += int(error != 0.0)
            self.errors_.append(errors)
        return self

class Perceptron3(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = np.zeros(5)
        self.errors_ = []

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, 0)

    def fit(self, X, y):

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):

                output = self.sigmoid(self.net_input(xi))
                error = target - output
                if target == 0:
                    error = -output
                elif target == 1:
                    error = 1 - output
                self.w_[1:] += self.eta * xi * error * output * (1 - output)
                self.w_[0] += self.eta * error * output * (1 - output)
                errors += int(self.predict(xi) != target)
            self.errors_.append(errors)
        return self

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)
print("Данные о цветках")
print(df.to_string())
df.to_csv("iris.csv")
X = df.iloc[0:100, [0, 2]].values
print(f"Значения Х {X}")
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
print("Значение названий цветков в виде -1 и 1, Y - 100")
print(y)
plt.scatter(X[0:50, 0], X[0:50, 1], color="red", marker="o", label="щетинистый")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="разноцветный")
plt.xlabel("длина чашелистника")
plt.ylabel("длина лепестка")
plt.legend(loc="upper left")
plt.show()
ppn = Perceptron(eta=0.001, n_iter=10)
ppn2 = Perceptron2(eta=0.001, n_iter=10)
ppn3 = Perceptron3(eta=0.001, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Эпохи")
plt.ylabel("Число случаев ошибочной классификации")
plt.show()



from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ("o", "x", "s", "^", "v")
    colors = ("red", "blue", "green", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    print(np.shape(X))

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.9, c=cmap(idx), marker=markers[idx], label=cl)


plot_decision_regions(X, y, classifier=ppn)
plt.show()
