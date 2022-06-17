import numpy as np
from typing import Optional
from sklearn.svm import SVC

from .draw_classification import draw


class SVM:
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
                 max_iter: Optional[int] = 500,
                 kernel: Optional[str] = 'linear',
                 degree: Optional[int] = 2,
                 draw_flag: Optional[bool] = False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.max_iter = max_iter
        self.kernel = kernel
        self.degree = degree
        self.draw_flag = draw_flag

    @property
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
        '''
        #self.X_train = np.concatenate((np.ones_like(X_train[:,0:1]), X_train), axis=1)
        max_feature_value = np.amax(self.y_train)
        features = self.X_train.shape[1]
        self.omega = np.array([max_feature_value]*features)
        print(self.omega.shape)
        learning_rate = [0.1**i*max_feature_value for i in range(3)]
        b_step_size = 2
        b_multiple = 5
        optimized = False
        for lr in learning_rate:
            while not optimized:
                for b in np.arange(-1*(max_feature_value*b_step_size), max_feature_value*b_step_size, lr*b_multiple):
                    for i in range(self.X_train.shape[0]):
                        if self.y_train[i]*(self.X_train[i]@self.omega.T+b)<1:
                            print(self.y_train[i],self.X_train[i], self.omega)
                            self.omega = np.add(self.omega,lr*(self.X_train[i]@self.y_train[i]*(-2)/features)*self.omega.T,self.omega, casting="unsafe")
                        else:
                            self.omega = np.add(self.omega,lr*((-2)/features)*self.omega.T,self.omega, casting="unsafe")
                            b_opt = b
                            optimized = True

                            print(self.omega,b_opt)
        print(self.X_test.shape,self.omega.shape)
        y_pred = np.sign(self.X_test@self.omega.T + b_opt)'''

        model = SVC(kernel=self.kernel, degree=self.degree, max_iter=self.max_iter)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test).T.reshape(-1, 1)
        if self.draw_flag:
            draw(X_test, y_test, y_pred).show()
        return np.concatenate((self.X_test, y_pred), axis=1)


if __name__ == "__main__":
    X = np.random.randint(100, size=(500, 2))
    # y = np.array([1 if i[0] > 10 and i[1] > 10 else 0 for i in X]).reshape((-1, 1))
    y = np.array([1 if (i[0] - 50) ** 2 + (i[1] - 50) ** 2 <= 600 else 2 for i in X]).reshape((-1, 1))
    X_train = X[:int(0.8*500), :]
    y_train = y[:int(0.8*500), :]
    X_test = X[int(0.8*500):, :]
    y_test = y[int(0.8*500):, :]
    pred = SVM(X_train, y_train, X_test, max_iter=100, draw_flag=1, kernel='poly', degree=6).solve
    print(pred, y_test)
