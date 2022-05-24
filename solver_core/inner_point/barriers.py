import numpy as np
from typing import Callable, Optional
import autograd.numpy as npa
from autograd import grad, hessian
from numpy.linalg import norm

from solver_core.inner_point.handlers.input_validation import check_expression, check_restr, check_point
from solver_core.inner_point.handlers.prepocessing import prepare_all


class LogBarrirers:
    """
       Метод решения задачи методом логарифмических барьеров.

       Parameters
       ----------
       function: Callable
           Функция для минимизации.

       restr_uneq: list
           Список вызываемых функций питона, которые имеют смысл левой части ограничений g(x)  0


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
        self.restrictions = restrictions
        self.x = x0
        self.eps = eps
        self.mu = 1

    def solve(self):
        l_hess = hessian(self.lagrange)
        l_grad = grad(self.lagrange)
        step = np.linalg.inv(l_hess(self.x)) @ l_grad(self.x)

        while not np.allclose(step, np.zeros(self.x.shape)):
            self.x = self.x - step
            step = np.linalg.inv(l_hess(self.x)) @ l_grad(self.x)
            self.mu *= 10
        return self.x

    def lagrange(self, x):
        l = self.function(x)*self.mu
        for i in self.restrictions:
            l -= npa.log(i(x))
        return l


if __name__ == '__main__':
    f = 'x1**2 + x2**2 + (0.5*1*x1 + 0.5*2*x2)**2 + (0.5*1*x1 + 0.5*2*x2)**4'
    subject_to = 'x1+x2<=0;2*x1-3*x2<=1'
    zakharov_point_min = np.array([0, 0])
    # zakharov_point_start = np.array([-5, 4.])
    zakharov_point_start = '-5;4'

    # input_validation
    f = check_expression(f)
    subject_to = check_restr(subject_to, method='log_barrier')
    zakharov_point_start = check_point(zakharov_point_start, f, subject_to, 'log_barrier')
    # preprocessing
    f, subject_to, zakharov_point_start = prepare_all(f, subject_to, 'log_barrier', zakharov_point_start)
    # solver
    task = LogBarrirers(f, subject_to, zakharov_point_start)
    ans = task.solve()
    print(np.allclose(ans, zakharov_point_min))
    print(ans)