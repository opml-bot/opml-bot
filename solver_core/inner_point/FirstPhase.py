from copy import deepcopy

import numpy as np
from typing import Callable, Optional
import autograd.numpy as npa
from autograd import grad, hessian
from numpy.linalg import norm

# from ..handlers.input_validation import check_expression, check_restr, check_point
# from ..handlers.prepocessing import prepare_all


class FirstPhase:
    """
       Метод решения задачи методом логарифмических барьеров.

       Parameters
       ----------
       n: int
           Число переменных.

       eps: Optional[float] = 1e-12
           Точность. От этого параметра зависит, когда будет остановлен итерационные процесс поиска решения. При
           слишком малых значениях, есть вероятность бесконечного цикла или nan в ответе.
       """

    def __init__(self,
                 n: int,
                 restrictions: list,
                 eps: Optional[float] = 1e-12):

        self.x = np.random.random(n)
        self.restrictions = restrictions
        self.s = max([i(self.x) for i in self.restrictions]) + 1
        self.eps = eps
        self.mu = 1


    def solve(self):

        for i in self.restrictions:
            if i(self.x) > 0:
                break
        else:
            return self.x
        l_hess = hessian(self.lagrange)
        l_grad = grad(self.lagrange)
        step = np.linalg.inv(l_hess(self.x)) @ l_grad(self.x)

        while self.s - 1. > 0:
            if np.isnan(self.x).any():
                raise ValueError('В ходе работы получился None.')
            self.x = self.x - step

            try:
                step = np.linalg.inv(l_hess(self.x)) @ l_grad(self.x)
            # -------------------------------------------------------
            except np.linalg.LinAlgError as e:
                for i in self.restrictions:
                    if i(self.x) > 0:
                        break
                else:
                    return np.round(self.x, 5)
            # -------------------------------------------------------
            self.s = max([i(self.x) for i in self.restrictions]) + 1
            self.mu += 1
        return np.round(self.x, 5)

    def lagrange(self, x):
        l = self.s*self.mu
        for i in self.restrictions:
            l -= npa.log(-(i(x) - self.s))
        return l


def rewrite(restr: Callable) -> Callable:
    """
    Функция переделывает ограничения с >= на <=.

    Parameters
    ----------
    restr: list
        Список питоновских функций, которые являются ограничениями для задачи (вида g(x) >= 0).

    Returns
    -------
    new: list
        Список переписанных функций (вида g(x) <= 0).
    """
    r = deepcopy(restr)
    new = lambda x: -r(x)
    return new


if __name__ == '__main__':
    subject_to = 'x1+x2<=0;2*x1-3*x2<=1'
    zakharov_point_min = [0, 0]
    # zakharov_point_start = np.array([-5, 4.])
    zakharov_point_start = '-5;4'

    f = lambda x: x[0] + x[1]
    consts = [lambda x: (-2*x[0] - 4*x[1] + 25), lambda x: (-x[0] + 8), lambda x: (-2*x[1] + 10),
              lambda x: x[0], lambda x: x[1]] # ограничения на >=
    consts = [rewrite(i) for i in consts]
    consts = [lambda x: x[0] + x[1], lambda x: 2*x[0] - 3*x[1] - 1]

    p = FirstPhase(2, consts)
    print(p.solve())
