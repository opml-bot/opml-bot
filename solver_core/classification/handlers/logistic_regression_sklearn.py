import numpy as np
from typing import Optional

import sklearn


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
                 degree: Optional[int] = 1,
                 regularization: Optional[bool] = False,
                 draw_flag: Optional[bool] = False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.regularization = regularization
        self.draw_flag = draw_flag
        self.degree = degree

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
        X_train_poly = sklearn.preprocessing.PolynomialFeatures(degree = self.degree).fit_transform(self.X_train)
        X_test_poly = sklearn.preprocessing.PolynomialFeatures(degree = self.degree).fit_transform(self.X_test)

        if self.regularization:
            model = sklearn.linear_model.LogisticRegression(penalty='l1')
        else:
            model = sklearn.linear_model.LogisticRegression(penalty='none')

        y_pred = model.fit(X_train_poly,self.y_train).predict(X_test_poly)
        omega = [model.intercept_, model.coef_]
        return np.concatenate((self.X_test, y_pred), axis=1), omega


if __name__ == "__main__":
    X = np.random.randint(21, size=(100, 2))
    y = np.array([1 if i[0] > 10 and i[1] > 10 else 0 for i in X]).reshape((-1, 1))
    X_train = X[:80, :]
    y_train = y[:80, :]
    X_test = X[80:, :]
    y_test = y[80:, :]
    pred, omega = LogisticRegression(X_train, y_train, X_test, degree = 1,regularization=True).solve()
    print(pred, y_test)
    print(omega)
