
import numpy as np
from typing import Optional
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures

from solver_core.classification.handlers.draw_classification import draw

class RadicalRegression:
    """
    Модель для логистической регрессии с радиальными базисными функциями. Находит ....
    Parameters
    ----------
    X_train: np.ndarray
        Массив обучающих данных. Может быть одномерными и многомерными.
    y_train: np.ndarray
        Массив значений обучающих данных. Строго одномерный.
    X_test: np.ndarray
        Массив данных, на которых тестируется модель. Может быть одномерными и многомерными.
    mu: Optional[float] = 0.5
        Критерий, по которому отделяются классы в логистической регрессии.
    max_iter: Optional[int] = 500
        Максимальное количество итераций алгоритма.
    delta_w: Optional[float] = 100
        Параметр остановки алгоритма. Как только разница между результатами соседним итераций меньше чем данный параметр, алгоритм прекращает работу.
    draw_flag: Optional[bool] = False
        Флаг рисования результатов работы модели. Принимает значения "True" или "False.
    """

    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 mu: Optional[float] = 0.5,
                 delta_w: Optional[float] = 100,
                 max_iter: Optional[int] = 500,
                 type: Optional[str] = 'linear',
                 draw_flag: Optional[bool] = False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.draw_flag = draw_flag
        self.omega = np.zeros(X_train.shape[1] + 1)
        self.delta_w = delta_w
        self.max_iter = max_iter
        self.mu = mu
        self.type = type

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
        if self.type != 'linear':
            poly = PolynomialFeatures(degree=self.degree)
            poly.fit_transform(self.X_train)
        self.X_train = np.concatenate((np.ones_like(X_train[:,0:1]), X_train), axis=1)
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
        rbf_feature = RBFSampler(gamma=1,n_components=len(self.omega[1:]), random_state=1)
        X_features = rbf_feature.fit_transform(self.X_test)
        z = self.omega[0] + X_features @ self.omega[1:]
        y_pred = 1 / (1 + np.exp(-z))
        mu = np.mean(y_pred.flatten())
        y_pred = np.array([1 if i[0] >= mu else 0 for i in y_pred]).reshape((-1, 1))
        
        if self.draw_flag:
            draw(X_test, y_test, y_pred).show()
        return np.concatenate((self.X_test, y_pred), axis=1)


if __name__ == "__main__":
    X = np.random.randint(21, size=(100, 2))
    y = np.array([1 if i[0] > 10 and i[1] > 10 else 0 for i in X]).reshape((-1, 1))
    X_train = X[:80, :]
    y_train = y[:80, :]
    X_test = X[80:, :]
    y_test = y[80:, :]
    pred = RadicalRegression(X_train, y_train, X_test).solve()
    print(pred, y_test)
