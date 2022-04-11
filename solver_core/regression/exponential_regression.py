import numpy as np

from typing import Optional
from scipy.optimize import minimize


class ExpRegression:
    """
    Модель для экспоненциальной регрессии. Находит необходимые коэфициенты.

    Parameters
    ----------
    X: np.ndarray
        Массив регрессеров. Строго двумерный (даже если регрессор один.

    y: np.ndarray
        Массив значений целевой функции. Строго одномерный.

    alpha: Optional[float] = 0.5
        Степень влияния регуляризации. На этот коэфициент домножается регуляризации при минимизации МНК.

    regularization: Optional[str] = None
        Тип регуляризации. Принимает значения "L1", "L2", либо "norm".

    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 alpha: Optional[float] = 0.5,
                 regularization: Optional[str] = None):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.regularization = regularization

        # omega - искомые коэфициенты. Здесь задаются начальные значения
        self.omega = np.zeros(X.shape[1] + 1)
        self.omega[0] = 1

    def solve(self):
        """
        Метод для запуска решения. В нем составляется МНК и он минимизируется с помощью scipy.

        Returns
        -------
        func: str
            Функция в аналитическом виде.

        koefs: np.ndarray
            Коэфициенты регрессии (при x).

        free_member: float
            Свободный член регресси.
        """

        if self.regularization is None:
            self.regularization = 'None'
        reg_types = {'L1': self.l1, 'L2': self.l2, 'None': lambda x: 0}
        reg = reg_types[self.regularization]
        # функция для оптимизации
        def to_optim(koef):
            """
            Функция от коэфициентов (которые необходимо найти). Здесь составляется МНК и добавляется регуляризация
            (если задана). Все это выраженние впоследствии минимизируется.

            Parameters
            ----------
            koef: np.ndarray
                Коэфициенты  регресии, которые нужно найти.

            Returns
            -------
            MNK: float
                Значение функции (МНК + регуляризация) при заданныъ коэфициентах.
            """

            MNK = 0
            for i in range(self.X.shape[0]):
                prediction = koef[0] * np.exp(np.sum(self.X[i, :].flatten() * koef[1:]))
                MNK += (self.y[i] - prediction) ** 2
            MNK += reg(koef)
            return MNK

        answer = minimize(to_optim, self.omega)
        self.omega = np.array(answer['x'])
        # подготовка ответа
        func = f'{self.omega[0]:.4f}*exp({self.omega[1]:.4f}*x1'
        if len(self.omega > 2):
            for i in range(1, len(self.omega) - 1):
                func += f'{self.omega[i+1]:+.4f}*x{i+1}'
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
        x: np.ndarray
            Значения x для предсказания. Должен быть двумерным (даже если скаляр).

        Returns
        -------
        prediction: np.ndarray
            Предсказанное значение.
        """

        prediction = self.omega[0] * np.exp(x @ self.omega[1:].reshape(-1, 1))
        return prediction.flatten()

    def r2(self):
        """
        Коэфициент детерминации. Используется для измерения точности построенной регресии.

        Returns
        -------
        score: float
            Значение от -inf до 1. Чем больше, тем точнее построенная регрессия.
        """

        score = 1 - np.sum((self.y - self.predict(self.X)) ** 2) / np.sum((self.Y - self.Y.mean()) ** 2)
        return score

    def l1(self, koef=None):
        """
        Регуляризация L1.

        Returns
        -------
        float
            Сумма коэфициентов по модулю, умноженная на параметр alpha (alpha задается пользователем).
        """

        return self.alpha * np.sum(np.abs(koef))

    def l2(self, koef=None):
        """
        Регуляризация L1.

        Returns
        -------
        float
            Сумма коэфициентов в квадрате, умноженная на параметр alpha (alpha задается пользователем).
        """

        return self.alpha * np.sum(koef**2)

    def norm(self, koef=None):
        # пока не готово)))
        return 1


if __name__ == "__main__":
    # generate random exp regression
    n_samples = 100
    n_features = 1
    noise = 10
    a = 100
    b = np.array([2])
    #b = np.array([0.2, 0.6, 0.1])
    # a = np.random.random()*100
    # b = np.random.random(n_features)*5

    X = np.random.random(n_samples * n_features).reshape((n_samples, n_features))
    Y = (a * np.exp(X @ b.reshape((-1, 1)))).flatten() + noise * np.random.random(n_samples)

    task = ExpRegression(X, Y)
    s = task.solve()
    y_pred = task.predict(X)
    x_ = np.array([2])
    print(x_)
    y_pred = task.predict(x_.reshape(-1, 1))
    print(y_pred)
