import pandas as pd
import numpy as np
from typing import Optional
import math
# from sklearn.datasets import make_classification
# from scipy.spatial.distance import pdist, squareform
# from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel,linear_kernel
# from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
# from sklearn.svm import LinearSVC



class Pegasos:
    """
    Класс для решения задачи классификации на два класса методом опорных векторов SVM с применением алгоритма
    градиентного спуска для минимизации функции ошибок.

    Parameters
    ----------
    X: np.ndarray
        Массив признаков.
    y: np.ndarray
        Массив правильных меток для признаков.
    reg: Optional[float] = 1
        Параметр регуляризации.
    T: Optional[int] = 500
        Максимальное число итераций.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 reg: Optional[float] = 1,
                 T: Optional[int] = 500):
        self.X = X
        self.y = y.flatten()
        self.lam = reg
        self.n_iter = T
        self.w = np.zeros(self.X.shape[1])

    def solve(self):
        """
        Метод решает задачу
        """

        for i in range(self.n_iter):
            k = np.random.randint(0, self.y.shape[0])
            xk = self.X[k].flatten()
            yk = self.y[k]
            nu = 1 / (self.lam + i)
            predict = np.sum(self.w*xk)
            c = 1 - yk*predict
            if c < 1:
                self.w = (1 - nu*self.lam)*self.w + nu*c*xk
            else:
                self.w = (1 - nu * self.lam)*self.w
        predict = [1 if np.sum(self.w*i) > 0 else -1 for i in self.X]
        acc = np.where(predict != self.y, 0, 1).sum()/y.shape[0]
        print(acc)
        return self.y, predict, self.w


if __name__ == "__main__":
    from solver_core.stochastic_optimization.handler import prepare_data

    X = np.array([1, 1, 1, 1, 1, 0, 0 ,0 ,0 ,0 ]).reshape((-1, 1))
    y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    X = np.random.randint(100, size=(2000, 2))
    # y = np.array([1 if i[0] > 5 and i[1] > 5 else 0 for i in X]).reshape((-1, 1))
    y = np.array([1 if (i[0] - 50) ** 2 + (i[1] - 50) ** 2 <= 600 else -1 for i in X]).reshape((-1, 1))

    df = pd.DataFrame(X)
    df['Y'] = y
    df.to_csv('data.csv', index=False)

    X, y, xest, ytest = prepare_data('data.csv')
    task = Pegasos(X, y, T=10**6)
    ans = task.solve()
