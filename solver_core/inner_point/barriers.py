import numpy as np
from typing import Callable, Optional
import autograd.numpy as npa
from autograd import grad, hessian
from numpy.linalg import norm

# from ..handlers.input_validation import check_expression, check_restr, check_point
# from ..handlers.prepocessing import prepare_all

from solver_core.inner_point.handlers.input_validation import *
from solver_core.inner_point.handlers.prepocessing import prepare_all

class LogBarrirers:
    """
       Метод решения задачи методом логарифмических барьеров.

       Parameters
       ----------
       function: Callable
           Функция для минимизации.

       restr_uneq: list
           Список вызываемых функций питона, которые имеют смысл левой части ограничений g(x) <= 0

       x0: np.ndarray
           Плоский массив, содержащий начальную точку.

       eps: Optional[float] = 1e-12
           Точность. От этого параметра зависит, когда будет остановлен итерационные процесс поиска решения. При
           слишком малых значениях, есть вероятность бесконечного цикла или nan в ответе.

       mu: float
           Множитель для увеличения mu на каждой итерации.
       """

    def __init__(self,
                 function: Callable,
                 restrictions: list,
                 x0: np.ndarray,
                 mu: Optional[float] = 2.,
                 eps: Optional[float] = 1e-12):
        self.function = function
        self.restrictions = restrictions
        self.x = x0
        self.eps = eps
        self.mu = 1
        self.m = mu

    def solve(self):
        l_hess = hessian(self.lagrange)
        l_grad = grad(self.lagrange)
        step = np.linalg.inv(l_hess(self.x)) @ l_grad(self.x)

        while not np.allclose(step, np.zeros(self.x.shape)):
            print('-----------------------ITER------------------------------------')
            print(self.x)
            self.x = self.x - step
            try:
                step = np.linalg.inv(l_hess(self.x)) @ l_grad(self.x)
            except np.linalg.LinAlgError as e:
                print('ERROR')
                return self.x
            if np.isnan(self.x).any():
                print('ERROR')
                return self.x
            self.mu *= self.m
        return self.x

    def lagrange(self, x):
        l = self.function(x)*self.mu
        for i in self.restrictions:
            l -= npa.log(-i(x))
        return l


if __name__ == '__main__':
    # f = 'x1**2 + x2**2 + (0.5*1*x1 + 0.5*2*x2)**2 + (0.5*1*x1 + 0.5*2*x2)**4'
    # subject_to = 'x1+x2<=0;2*x1-3*x2<=1'
    # zakharov_point_min = np.array([0, 0])
    # # zakharov_point_start = np.array([-5, 4.])
    # zakharov_point_start = '-5;4'
    # zakharov_point_start = ''

    f = '3*x1 + 5*x2'
    subject_to = 'x1<=3; 2*x2<=12; 3*x1+2*x2<=18; x1>=0; x2>=0'
    zakharov_point_min = (2, 6)
    # zakharov_point_start = '2; 6'
    zakharov_point_start = ''

    f = "-2*x1-x2"
    subject_to = "-1*x1-0.1*x2<=-1; -1*x1+0.6*x2<=-1; -0.2*x1+1.5*x2<=-0.2; 0.7*x1+0.7*x2<=0.7; 2*x1-0.2*x2<=2; 0.5*x1-1*x2<=0.5; -1*x1-1.5*x2<=-1"
    zakharov_point_min = (4.7, 3.5)
    zakharov_point_start = ''
    # input_validation
    f = check_expression(f)
    subject_to = check_restr(subject_to, method='log_barrier')
    zakharov_point_start = check_point(zakharov_point_start, f, subject_to, 'log_barrier')
    # preprocessing
    f, subject_to, zakharov_point_start = prepare_all(f, subject_to, 'log_barrier', zakharov_point_start)
    # solver
    print(zakharov_point_start)
    for i in subject_to:
        print(i(zakharov_point_start), i(zakharov_point_start) <= 0)
    task = LogBarrirers(f, subject_to, zakharov_point_start, mu=2)
    ans = task.solve()
    print(np.allclose(ans, zakharov_point_min))
    print(ans)