import numpy as np
from typing import Optional


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

    mu: Optional[float] = 0.5
        Критерий, по которому отделяются классы в логистической регрессии.

    max_iter: Optional[int] = 500
        Максимальное количество итераций алгоритма.

    delta_w: Optional[float] = 100
        Параметр остановки алгоритма. Как только разница между результатами соседним итераций меньше чем данный параметр, алгоритм прекращает работу.

    alpha: Optional[float] = 0.5
        Степень влияния регуляризации. На этот коэфициент домножается регуляризация.

    regularization: Optional[bool] = False
        Флаг применения регуляризации. Принимает значения "True" или "False.

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
        self.mu = mu

    def solve(self):
        """
        Метод для запуска решения. В нем логичтическая регрессия.

        Returns
        -------
        func: str
            Массив предсказанных классов в формате [x_new|t_new], где t - класс.

        koefs: np.ndarray
            Коэфициенты регрессии w.

        """
        self.X_train = np.concatenate((np.ones_like(X_train), X_train), axis=1)
        self.omega = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
        # print(self.omega)
        i = 0
        old_w = np.array([-(10 ** 3)] * len(self.omega))
        while np.sum((old_w - self.omega) ** 2) > self.delta_w and i < self.max_iter:
            # print('norma',np.sum((old_w - self.omega) ** 2))
            z = self.X_train @ self.omega
            # print('z=',z)
            p = 1 / (1 + np.exp(-z))
            # print('p=',p)
            g = p * (1 - p)
            # print('g=',g)
            u = z + (self.y_train - p) / g
            # print('u=',u)
            G = np.zeros((len(g), len(g)), float)
            np.fill_diagonal(G, g)
            # print('G=',G)
            E = np.zeros((g.shape[1], g.shape[1]), int)
            np.fill_diagonal(E, 1)
            old_w = self.omega
            # print('X_train=',self.X_train.shape)
            if self.regularization:
                self.omega = np.linalg.inv(
                    self.X_train.T @ G @ self.X_train + self.alpha * E) @ self.X_train.T @ G @ u
            else:
                self.omega = np.linalg.inv(self.X_train.T @ G @ self.X_train) @ self.X_train.T @ G @ u
            i += 1

        # print('omega = ',self.omega)
        # print('X_test',self.X_test )
        z = self.omega[0] + self.X_test @ self.omega[1:]
        # print('z_last',z)
        y_pred = 1 / (1 + np.exp(-z))
        # print('y_pred',y_pred)
        mu = np.mean(y_pred.flatten())
        print('mu = ', mu)
        # print(y_pred)
        y_pred = np.array([1 if i[0] >= mu else 0 for i in y_pred]).reshape((-1, 1))
        # print(y_pred)
        # print('omega', self.omega)
        # print('res', np.concatenate((self.X_test, y_pred), axis=1))
        return np.concatenate((self.X_test, y_pred), axis=1), self.omega


if __name__ == "__main__":
    X = np.random.randint(21, size=(100, 1))
    y = np.array([1 if i > 10 else 0 for i in X]).reshape((-1, 1))
    X_train = X[:80, :]
    y_train = y[:80, :]
    X_test = X[80:, :]
    y_test = y[80:, :]
    pred, omega = LogisticRegression(X_train, y_train, X_test, regularization=True).solve()
    print(pred, y_test)
    print(omega)
