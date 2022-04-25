import numpy as np
from typing import Callable, Optional
from math import isclose
# import autograd.numpy as npa
from autograd import grad, hessian

from solver_core.inner_point.handlers.prepocessing import prepare_func, prepare_constraints, get_variables


class PrimalDual:
    """
    Метод решения задачи Прямо-двойственным методом внутренней точки.

    Parameters
    ----------

    """

    def __init__(self, function, restrictions, x0, k=0.8):
        self.function = function
        self.restrictions = restrictions
        self.k = k
        self.x = x0
        self.lam = np.array([0.5 for i in range(len(restrictions))])

        self.hes_L = hessian(self.function)
        self.hes_g = [hessian(i) for i in self.restrictions]
        self.grad_g = [grad(i) for i in self.restrictions]
        self.grad_L = grad(self.function)
        self.mu = sum([self.lam[i] * self.restrictions[i](self.x) for i in range(len(self.restrictions))])
        print(self.mu)
        self.k = 0.9


    def solve(self):
        nabla_L = self.grad_L(self.x)
        G = []
        for i in range(len(self.restrictions)):
            nabla_L -= self.lam[i] * self.grad_g[i](self.x)
            G.append(self.restrictions[i](self.x))
        G = np.diag(G)
        iter = 0
        while not isclose(max(sum([i**2 for i in nabla_L]), np.sqrt(np.sum([G.diagonal()[i]*self.lam[i]**2 for i in range(len(self.lam))]))), 0, abs_tol=1e-7):
            print(max(sum([i**2 for i in nabla_L]), np.sqrt(np.sum([G.diagonal()[i]*self.lam[i]**2 for i in range(len(self.lam))]))))
            hessian_of_lagrangian = self.hes_L(self.x)
            G = []
            nabla_L = self.grad_L(self.x)
            for i in range(len(self.restrictions)):
                hessian_of_lagrangian -= self.hes_g[i](self.x) * self.lam[i]
                if i == 0:
                    nabla_g = -(self.grad_g[i](self.x)).reshape((-1, 1))
                else:
                    nabla_g = np.hstack((nabla_g, (-self.grad_g[i](self.x)).reshape((-1, 1))))
                G.append(self.restrictions[i](self.x))
                nabla_L -= self.lam[i] * self.grad_g[i](self.x)
            Lambda_g = np.diag(self.lam) @ (-nabla_g.T)
            Lambda_G = -np.diag(self.lam) @ np.array(G).reshape((-1, 1))
            G = np.diag(np.array(G))
            matr = np.vstack((np.hstack((hessian_of_lagrangian, nabla_g)), np.hstack((Lambda_g, G))))
            nabla_L = nabla_L.reshape((-1, 1))
            Lambda_G = Lambda_G + self.mu
            Lambda_G = -Lambda_G
            right_vec = np.vstack((nabla_L, Lambda_G))

            ans = np.linalg.solve(matr, right_vec).flatten()
            x = -ans[:len(ans)//2]
            lam = -ans[len(ans)//2:]
            # поиск длины шага
            alpha_d = min([1] + [-self.k*self.lam[i]/delta for i, delta in enumerate(lam) if delta < 0])
            alpha_p = 1
            while min([i(self.x + alpha_p * np.abs(x)) for i in self.restrictions]) <= 0:
                alpha_p = alpha_p/2

            self.x = self.x + alpha_p * x
            self.lam = self.lam + alpha_d*lam
            self.mu = self.mu/10
            print(f'mu: {self.mu: .5E} xk: ({self.x[0]:3.5f}, {self.x[1]:.5f}) alpha_d: {alpha_d:.4f} lambda_k: ({self.lam[0]:.5f}, {self.lam[1]:.5f})')
            iter += 1
        return self.x, self.lam



def gradient(function: Callable,
             x0: np.ndarray,
             delta_x=10**-8) -> np.ndarray:
    """
    Численно вычисляет градиент. Параметр  delta_x отвечает за шаг изменения аргумента в проивзводной.

    Parameters
    ----------
    function: Callable
        Функция от которой берут гралиент в смысле питоновской фунции.

    x0: np.ndarray
        Точка, в которой вычисляют градиент

    delta_x: Optional[float] = 1e-8
         Шаг для производной.

    Returns
    -------
    grad: np.ndarray
        Значения градиента в точке x
    """
    grad = []
    for i in range(len(x0)):
        delta_x_vec_plus = x0.copy()
        delta_x_vec_minus = x0.copy()
        delta_x_vec_plus[i] += delta_x
        delta_x_vec_minus[i] -= delta_x
        grad_i = (function(delta_x_vec_plus) - function(delta_x_vec_minus)) / (2 * delta_x)
        grad.append(grad_i)

    grad = np.array(grad)
    return grad


def hessian1(function: Callable,
             x0: np.ndarray,
             delta_x=10**-8) -> np.ndarray:
    """
    Численно вычисляет градиент. Параметр  delta_x отвечает за шаг изменения аргумента в проивзводной.

    Parameters
    ----------
    function: Callable
        Функция от которой берут гралиент в смысле питоновской фунции.

    x0: np.ndarray
        Точка, в которой вычисляют градиент

    delta_x: Optional[float] = 1e-8
         Шаг для производной.

    Returns
    -------
    grad: np.ndarray
        Значения градиента в точке x
    """

    hes = [[0 for j in range(len(x0))] for i in range(len(x0))]
    for i in range(len(x0)):
        for j in range(min(i+1, len(x0))):
            delta_two_xs = x0.copy()
            delta_two_xs[i] += delta_x
            delta_two_xs[j] += delta_x
            delta_xi = x0.copy()
            delta_xi[i] += delta_x
            delta_xj = x0.copy()
            delta_xj[j] += delta_x

            hes[i][j] = (function(delta_two_xs) - function(delta_xi) - function(delta_xj) + function(x0)) / (delta_x**2)
    hes = np.array(hes, dtype=np.float64)
    hes = hes+hes.T - np.diag(hes.diagonal())
    return hes


if __name__ == "__main__":
    f = '-x1 - x2 + x3 + x4'
    subject_to = ['x1 >= x2**2 - 1', 'x2>=0']
    vars = get_variables(f)

    f = prepare_func(f, vars)
    con = prepare_constraints(subject_to, vars, 'primal-dual')

    task = PrimalDual(f, con, np.array([0.5, 0.5]))
    ans = task.solve()
    print(ans )
