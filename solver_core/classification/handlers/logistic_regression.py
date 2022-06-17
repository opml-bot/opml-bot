import numpy as np
from typing import Optional

from sklearn.preprocessing import PolynomialFeatures

from solver_core.classification.handlers.draw_classification import draw


class LogisticRegression:
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

    max_iter: Optional[int] = 500
        Максимальное количество итераций алгоритма.

    delta_w: Optional[float] = 100
        Параметр остановки алгоритма. Как только разница между результатами соседним итераций меньше чем данный параметр, алгоритм прекращает работу.

    alpha: Optional[float] = 0.5
        Степень влияния регуляризации. На этот коэфициент домножается регуляризация.

    regularization: Optional[bool] = False
        Флаг применения регуляризации. Принимает значения "True" или "False.

    type: Optional[str] = 'linear'
        Тип классификации: линейная или полиномиальная. Принимает значения 'linear' и 'poly'.

    degree: Optional[int] = 1
        Показатель степени полиномиальной регрессии.

    draw_flag: Optional[bool] = False
        Флаг рисования результатов работы модели. Принимает значения "True" или "False.

    """

    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 mu: Optional[float] = 0.5,
                 alpha: Optional[float] = 0.5,
                 delta_w: Optional[float] = 100,
                 max_iter: Optional[int] = 500,
                 regularization: Optional[bool] = False,
                 type: Optional[str] = 'linear',
                 degree: Optional[int] = 1,
                 draw_flag: Optional[bool] = False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.regularization = regularization
        self.draw_flag = draw_flag
        self.omega = np.zeros(X_train.shape[1] + 1)
        self.delta_w = delta_w
        self.max_iter = max_iter
        self.alpha = alpha
        self.degree = degree
        self.type = type

    def solve(self):
        """
        Метод для запуска решения. В нем логичтическая регрессия.

        Returns
        -------
        func: str
            Массив предсказанных классов в формате [x_new|t_new], где t - класс.

        """
        if self.type != 'linear':
            poly = PolynomialFeatures(degree=self.degree)
            self.X_train = poly.fit_transform(self.X_train)
        self.X_train = np.concatenate((np.ones_like(X_train[:, 0:1]), X_train), axis=1)
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
            E = np.zeros((g.shape[1], g.shape[1]), int)
            np.fill_diagonal(E, 1)
            old_w = self.omega
            if self.regularization:
                self.omega = np.linalg.inv(
                    self.X_train.T @ G @ self.X_train + self.alpha * E) @ self.X_train.T @ G @ u
            else:
                self.omega = np.linalg.inv(self.X_train.T @ G @ self.X_train) @ self.X_train.T @ G @ u
            i += 1

        z = self.omega[0] + self.X_test @ self.omega[1:]
        y_pred = 1 / (1 + np.exp(-z))
        mu = np.mean(y_pred.flatten())
        y_pred = np.array([1 if i[0] >= mu else 0 for i in y_pred]).reshape((-1, 1))
        if self.draw_flag:
            draw(X_test, y_test, y_pred).show()
        return np.concatenate((self.X_test, y_pred), axis=1)


if __name__ == "__main__":
    X = np.random.randint(100, size=(500, 2))
    #y = np.array([1 if i[0] > 5 and i[1] > 5 else 0 for i in X]).reshape((-1, 1))
    y = np.array([1 if (i[0]-50)**2+(i[1]-50)**2 <= 600 else 0 for i in X]).reshape((-1, 1))
    X_train = X[:int(0.8*500), :]
    y_train = y[:int(0.8*500), :]
    X_test = X[int(0.8*500):, :]
    y_test = y[int(0.8*500):, :]
    pred = LogisticRegression(X_train, y_train, X_test,\
                              regularization=True,type='poly', degree=2,  draw_flag=1).solve()
    print(pred, y_test)
#