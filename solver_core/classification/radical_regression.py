import numpy as np
from typing import Optional

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

from .draw_classification import draw


class RadicalRegression:
    """
    Модель для логистической регрессии. Находит ....

    Parameters
    ----------
    X_train: np.ndarray
        Массив обучающих данных. Может быть одномерными и многомерными.

    y_train: np.ndarray
        Массив значений обучающих данных. Строго одномерный.

    X_test: np.ndarray
        Массив данных, на которых тестируется модель. Может быть одномерными и многомерными.

    y_test: np.ndarray
        Массив данных, верных значений тестовой выборки. Строго одномерный.

    max_iter: Optional[int] = 500
        Максимальное количество итераций алгоритма.

    delta_w: Optional[float] = 100
        Параметр остановки алгоритма. Как только разница между результатами соседним итераций меньше чем данный параметр, алгоритм прекращает работу.

    alpha: Optional[float] = 0.5
        Степень влияния регуляризации. На этот коэфициент домножается регуляризация.

    regularization: Optional[bool] = False
        Флаг применения регуляризации. Принимает значения "True" или "False.

    type_cl: Optional[str] = 'linear'
        Тип классификации: линейная или полиномиальная. Принимает значения 'linear' и 'poly'.

    degree: Optional[int] = 1
        Показатель степени полиномиальной регрессии.

    c: Optional[float] = 0,
        Точка, до которой расчитывается норма в РБФ.

    draw_flag: Optional[bool] = False
        Флаг рисования результатов работы модели. Принимает значения "True" или "False.

    """

    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 delta_w: Optional[float] = 100,
                 max_iter: Optional[int] = 500,
                 type_cl: Optional[str] = 'linear',
                 degree: Optional[int] = 1,
                 c: Optional[float] = None,
                 draw_flag: Optional[bool] = False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.draw_flag = draw_flag
        self.omega = np.zeros(X_train.shape[1] + 1)
        self.delta_w = delta_w
        self.max_iter = max_iter
        self.type_cl = type_cl
        self.c = c
        self.degree = degree


    def solve(self):
        """
        Метод для запуска решения. В нем регрессия.
        Returns
        -------
        func: str
            Массив предсказанных классов в формате [x_new|t_new], где t - класс.
        koefs: np.ndarray
            Коэфициенты регрессии w.
        """
        X_test_old = self.X_test
        if self.type_cl != 'linear':
            poly = PolynomialFeatures(degree=self.degree)
            self.X_train = poly.fit_transform(self.X_train)
            self.X_test = poly.fit_transform(self.X_test)
        else:
            self.X_train = np.concatenate((np.ones_like(self.X_train[:, 0:1]), self.X_train), axis=1)
            self.X_test = np.concatenate((np.ones_like(self.X_test[:, 0:1]), self.X_test), axis=1)
        self.omega = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
        i = 0
        old_w = np.array([-(10 ** 3)] * len(self.omega))
        while np.sum((old_w - self.omega) ** 2) > self.delta_w and i < self.max_iter:
            z = self.X_train @ self.omega
            p = 1 / (1 + np.exp(-z))
            g = p * (1 - p)
            u = z + (self.y_train - p) / g
            G = np.zeros((len(g), len(g)), float)
            np.fill_diagonal(G, g)
            old_w = self.omega
            self.omega = np.linalg.inv(self.X_train.T @ G @ self.X_train) @ self.X_train.T @ G @ u
            i += 1
        if self.c == None:
            self.c = minimize(lambda c: ((self.X_test-c)**2).sum()**0.5,[100]*self.X_test.shape[1]).x[0]
        r = np.array(((self.X_test[:,1:] - self.c)**2)**0.5)
        rbf = np.exp(-(0.000001 * r) ** 2)
        z = self.omega[0] + rbf @ self.omega[1:]
        y_pred = 1 / (1 + np.exp(-z))
        mu = np.mean(y_pred.flatten())
        y_pred = np.array([1 if i[0] >= mu else 0 for i in y_pred]).reshape((-1, 1))

        if self.draw_flag:
            draw(X_test_old, self.y_test, y_pred).to_html()
        return np.concatenate((self.X_test, y_pred), axis=1)


if __name__ == "__main__":
    X = np.random.randint(100, size=(500, 2))
    # y = np.array([1 if i[0] > 5 and i[1] > 5 else 0 for i in X]).reshape((-1, 1))
    y = np.array([1 if (i[0] - 50) ** 2 + (i[1] - 50) ** 2 <= 600 else 0 for i in X]).reshape((-1, 1))
    X_train = X[:int(0.8 * 500), :]
    y_train = y[:int(0.8 * 500), :]
    X_test = X[int(0.8 * 500):, :]
    y_test = y[int(0.8 * 500):, :]
    pred = RadicalRegression(X_train, y_train, X_test, y_test,type='poly',degree=2, draw_flag=1, c=0).solve()
    print(pred, y_test)
