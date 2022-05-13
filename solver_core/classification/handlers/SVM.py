import numpy as np
from typing import Optional
from sklearn.svm import SVC


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
                 X_test: np.ndarray):
        # learning_rate: Optional[float] = 0.01,
        # max_iter: Optional[int] = 500)\

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        # self.max_iter = max_iter
        # self.learning_rate = learning_rate

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

        model = SVC(kernel='linear')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print(y_pred.T.reshape(-1,1).shape)
        print(self.X_test.shape)
        print(model.intercept_,model.coef_[0])
        omega = model.intercept_.tolist()+ model.coef_[0].tolist()
        return np.concatenate((self.X_test, y_pred.T.reshape(-1,1)), axis=1), omega


if __name__ == "__main__":
    X = np.random.randint(21, size=(100, 2))
    y = np.array([1 if i[0] > 10 and i[1] > 10 else 0 for i in X]).reshape((-1, 1))
    X_train = X[:80, :]
    y_train = y[:80, :]
    X_test = X[80:, :]
    y_test = y[80:, :]
    pred, omega = SVM(X_train, y_train, X_test).solve
    print(pred, y_test)
    print(omega)
