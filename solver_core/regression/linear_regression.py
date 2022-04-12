# -*- coding: utf-8 -*-
"""linear_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7GOIywyHv2s-sntqpRBiQjhVUeWIZKG
"""

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

        # omega - искомые коэфициенты. Здесь задаются начальные значения
        if len(self.x_points.shape) == 1:
            self.omega = np.zeros(2)
        else:
            self.omega = np.zeros(self.x_points.shape[1] + 1)
        self.omega[0] = 1        

    def solve(self) -> list:
        """
        Метод для запуска решения. 
        
        Returns
        -------
        func: str
            Функция в аналитическом виде.
        koefs: np.ndarray
            Коэфициенты регрессии (при x).
        free_member: float
            Свободный член регресси.
        """

        if len(self.x_points.shape) == 1:
            self.X = np.array([np.ones(self.x_points.shape[0]), self.x_points]).T
        else:
            self.X = np.ones((self.x_points.shape[0],self.x_points.shape[1]+1))
            self.X[:,1:] = self.x_points


        if self.regularization is None:
            self.omega = np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.y_points)

        if self.regularization == 'l1':
            learning_rate = 0.001
            l1 = self.alpha
            self.omega = np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.y_points)
            for t in range(500):
                pred = np.dot(self.omega, self.X.T)
                delta = pred - self.y_points
                self.omega = self.omega - learning_rate * (self.X.T.dot(delta) + l1 * np.sign(self.omega))

        if self.regularization == 'l2':
            l2 = self.alpha * 100
            self.omega = np.dot(np.linalg.inv(l2 * np.eye(2) + self.X.T.dot(X)), np.dot(self.X.T, self.y_points))

        if self.regularization == 'norm':
            self.omega = np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.y_points)
            e = np.random.normal(size=self.X.shape[0])
            pred = np.dot(self.omega, self.X.T) + e
            model = sm.OLS(pred, self.X)
            results = model.fit()
            self.omega = results.params
            
        # подготовка ответа
        func = f'{self.omega[0]:.4f} + {self.omega[1]:.4f}*x1'
        if len(self.omega > 2):
            for i in range(1, len(self.omega) - 1):
                func += f' {self.omega[i+1]:+.4f}*x{i+1}'
        func += ')'
        
        koefs = self.omega[1:]
        free_member = self.omega[0]
        return func, koefs, free_member

    def predict(self, x):
        """
        Метод получает на вход значение/значения X и на основе полученных коэфициентов регресии предсказывает значение.
        По сути подставляет в полученную функцию иксы.
        Parameters
        ----------
        x: np.ndarray or float
            Значения x для предсказания. Может быть числом или np.array размерности 2 или 1.
        Returns
        -------
        prediction: np.ndarray
            Предсказанное значение.
        """
        if len(x.shape) == 1:
            X = np.array([np.ones(x.shape[0]), x]).T
        else:
            X = np.ones((x.shape[0],x.shape[1]+1))
            X[:,1:] = x
        prediction = X.dot(self.omega)
        return prediction.flatten()

    def r2(self):
        """
        Коэфициент детерминации. Используется для измерения точности построенной регресии.
        Returns
        -------
        score: float
            Значение от -inf до 1. Чем больше, тем точнее построенная регрессия.
        """

        score = 1 - np.sum((self.y_points - self.predict(self.x_points)) ** 2) / np.sum((self.y_points - self.y_points.mean()) ** 2)
        return score

if __name__ == '__main__':
    X = np.sort(np.random.choice(np.linspace(0, 2 * np.pi, num=1000), size=25, replace=True))
    y = np.sin(X) + 1 + np.random.normal(0, 0.3, size=X.shape[0])

    task = LinearRegression(X=X, y=y,regularization='norm')
    answer = task.solve()
    print(answer)
    y_pred = task.predict(X)
    print(y_pred)
    print(task.r2())
