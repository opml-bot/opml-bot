import numpy as np
from typing import Callable, Optional
from scipy.optimize import brent
from math import isclose
import autograd.numpy as npa
from autograd import grad, hessian
from numpy.linalg import norm

from solver_core.inner_point.handlers.input_validation import check_expression, check_restr, check_point
from solver_core.inner_point.handlers.prepocessing import prepare_all


class Newton:
    """
    Метод решения задачи методом Ньютона.

    Parameters
    ----------
    function: Callable
        Функция для минимизации.

    restrictions: list
        Список из двух массивов numpy: A, b. A - матрица для линейных ограничений (то что слева), b - массив
        коэфициентов.

    x0: np.ndarray
        Плоский массив, содержащий начальную точку.

    eps: Optional[float] = 1e-12
        Точность. От этого параметра зависит, когда будет остановлен итерационные процесс поиска решения. При
        слишком малых значениях, есть вероятность бесконечного цикла или nan в ответе.
    """

    def __init__(self,
                 function: Callable,
                 restrictions: list,
                 x0: np.ndarray,
                 eps: Optional[float] = 1e-12):
        self.function = function
        self.A = restrictions[0]
        self.b = restrictions[1]
        self.x = x0
        self.eps = eps

        self.hess = hessian(function)
        self.grad = grad(function)
        self.lam = np.array([0.5 for i in range(len(restrictions))])
        self.mu = 1
        self.N_con = len(restrictions)

    def solve(self):
        """
        Метод решает задачу.

        Returns
        -------
        x: np.ndarray
            Массив с решениями прямой задачи.

        lam: np.ndarray
            Массив с решениями двойственной задачи.
        """
        for _ in range(1000):
            if (np.isnan(self.x)).any():
                print(self.x)
                break
            W = self.hess(self.x)
            g = self.grad(self.x)
            matr = np.vstack((np.hstack((W, self.A.T)), np.hstack((self.A, np.zeros(shape=W.shape)))))
            vec = -np.vstack(((g + self.A@self.lam).reshape(-1, 1), (self.A@self.x - self.b).reshape(-1, 1)))
            r = np.linalg.solve(matr, vec)
            # new_x = r[:W.shape[0]].flatten()
            # new_lam = r[W.shape[0]:].flatten()
            new_x = r[W.shape[0]:].flatten()
            new_lam = r[:W.shape[0]].flatten()

            def func(alpha, X=0, Lam=0):
                g = self.grad(self.x + alpha*X)
                vec = np.vstack(((g + self.A @ (self.lam + alpha*Lam)).reshape(-1, 1),
                                  (self.A @ (self.x + X) - self.b).reshape(-1, 1)))
                return norm(vec)
            alpha = brent(func, brack=(0, 1), args=(new_x, new_lam))
            alpha = abs(alpha)
            print(alpha)
            while ((self.lam + alpha * new_lam) < 0).any():
                alpha /= 2
            self.x += alpha*new_x
            self.lam += alpha * new_lam
            print(self.x)


    def calc_constrains(self, x):
        """
        Метод подставляет x в функции ограничения.

        Parameters
        ----------
        x: np.ndarray
            Значения точки для подстановки в ограничения.

        Returns
        -------
        ans: np.ndarray
            Массив со значениями функций ограничений.
        """
        ans = np.array([self.restrictions[i](x) for i in range(self.N_con)])
        return ans

    def minimize_func(self, x):
        """
        Метод вычисляет значения функци вместе с барьерной функцией (Лагранжиан).

        Returns
        -------
        r: float
            Значения указанной в описании функции.
        """
        r = self.function(x)
        for i in self.restrictions:
            r -= self.mu*npa.log(i(x))
        return r

    def minimize_func_grad(self):
        """
        Метод вычисляет значения градиента Лагранжиана.

        Returns
        -------
        g : float
            Значения указанной в описании функции.
        """
        g = self.grad_f(self.x)
        grads = [i(self.x) for i in self.grad_g]
        for i in range(self.N_con):
            g -= self.mu*grads[i]
        return g


if __name__ == '__main__':
    f = 'x1**2 + x2**2 + (0.5*1*x1 + 0.5*2*x2)**2 + (0.5*1*x1 + 0.5*2*x2)**4'
    subject_to = 'x1+x2=0;2*x1-3*x2=0'
    zakharov_point_min = [0, 0]
    # zakharov_point_start = np.array([-5, 4.])
    zakharov_point_start = '-5;4'

    # input_validation
    f = check_expression(f)
    subject_to = check_restr(subject_to, method='Newton')
    zakharov_point_start = check_point(zakharov_point_start, f, subject_to, 'Newton')
    # preprocessing
    f, subject_to, zakharov_point_start = prepare_all(f, subject_to, 'Newton', zakharov_point_start)
    # solver
    task = Newton(f, subject_to, zakharov_point_start)
    ans = task.solve()
    print(ans)

