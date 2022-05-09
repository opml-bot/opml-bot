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
                 alpha: Optional[float] = 0.5,
                 delta_w: Optional[float] = 0.001,
                 lam: Optional[float] = 0.001,
                 regularization: Optional[bool] = False,
                 draw_flag: Optional[bool] = False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.regularization = regularization
        self.draw_flag = draw_flag
        self.omega = np.zeros(X_train.shape[1] + 1)
        self.delta_w = delta_w
        self.alpha = alpha

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
        self.omega = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
        for i in range(len(self.omega)):
            if (self.omega[i + 1] - self.omega[i]) ** 2 > self.delta_w:
                z = self.X_train @ self.omega
                p = 1 / (1 + np.exp(-z))
                g = p * (1 - p)
                u = z + (self.y_train - p) / g
                G = np.zeros((len(g), len(g)), float)
                np.fill_diagonal(G, g)
                E = np.zeroes((len(g), len(g)), int)
                np.fill_diagonal(E, 1)
                if self.regularization:
                    self.omega = np.linalg.inv(self.X_train.T @ G @ self.X_train + self.alpha@E) @ self.X_train @ G @ u
                else:
                    self.omega = np.linalg.inv(self.X_train.T @ G @ self.X_train) @ self.X_train @ G @ u

        z = self.omega[0] + self.X_test@self.omega[1:]
        y_pred = 1 / (1 + np.exp(-z))
        return np.concatenate((self.X_test,y_pred),axis=1), self.omega

if __name__ == "__main__":

