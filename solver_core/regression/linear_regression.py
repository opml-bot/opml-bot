from typing import Optional

import numpy as np
from sklearn import linear_model
import statsmodels.api as sm


class LinearRegression:
    """
    Обычная модель линейной регрессии.

    Parameters
    ----------
    X : np.ndarray
        Тренировочные данные.
    y : np.ndarray
        Значения целевой функции.
    regularization : Optional[str] = None
        Тип регуляризации.
    alpha : Optional[int] = 10
        Скорость обучения.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 regularization: Optional[str] = None,
                 alpha: Optional[int] = 10):

        self.x_points = X
        self.y_points = y
        self.regularization = regularization
        self.alpha = alpha

    def solve(self) -> list:
        """

        Returns
        -------
        list
            Список, содержащий функцию, предсказанные значения и свободный коэффициент.
        """

        X = np.array([np.ones(self.x_points.shape[0]), self.x_points]).T

        if self.regularization is None:
            w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), self.y_points)
            func = 'y= ' + '{:.3f}'.format(w[1]) + '*x +' + '{:.3f}'.format(w[0])
            predict = np.dot(w, X.T)
            free_member = float('{:.3f}'.format(float(w[0])))

        if self.regularization == 'l1':
            learning_rate = 0.001
            l1 = self.alpha
            w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), self.y_points)
            for t in range(500):
                predict = np.dot(w, X.T)
                delta = predict - self.y_points
                w = w - learning_rate * (X.T.dot(delta) + l1 * np.sign(w))
            func = f'y = {w[0]} + {w[1]} * x' if w[1] >= 0 else f'y = {w[0]} - {abs(w[1])} * x'
            predict = np.dot(w, X.T)
            free_member = float('{:.3f}'.format(float(w[0])))

        if self.regularization == 'l2':
            l2 = self.alpha * 100
            w = np.dot(np.linalg.inv(l2 * np.eye(2) + X.T.dot(X)), np.dot(X.T, self.y_points))
            func = f'y = {w[0]} + {w[1]} * x' if w[1] >= 0 else f'y = {w[0]} - {abs(w[1])} * x'
            predict = X.dot(w)
            free_member = float('{:.3f}'.format(float(w[0])))

        if self.regularization == 'norm':
            w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), self.y_points)
            e = np.random.normal(size=len(w))
            predict = np.dot(w, X.T)
            model = sm.OLS(predict, X)
            results = model.fit()
            w = results.params
            func = f'y = {w[0]} + {w[1]} * x' if w[1] >= 0 else f'y = {w[0]} - {abs(w[1])} * x'
            free_member = float('{:.3f}'.format(float(w[0])))
        return [func, predict, free_member]


if __name__ == '__main__':
    X = np.sort(np.random.choice(np.linspace(0, 2 * np.pi, num=1000), size=25, replace=True))
    y = np.sin(X) + 1 + np.random.normal(0, 0.3, size=X.shape[0])

    task = LinearRegression(X=X, y=y, regularization='norm')
    answer = task.solve()
    print(answer)
