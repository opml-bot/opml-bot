import autograd.numpy as np
from scipy.optimize import minimize
from autograd import jacobian

# from solver_core.solver_gradient.handlers.preprocessing import prepare_func_newton, get_variables


class NewtonCG:
    """
    Класс для решения задачи многомерной оптимизации методом Ньютон-сопряженного градиента. За основу взята функция из
    scipy и функция нахождения матрицы
    """
    def __init__(self, function, x0, jac=None):
        self.function = function
        self.x0 = np.array(list(x0))
        self.jac = jac

    def solve(self):
        if not self.jac:
            self.jac = jacobian(self.function)
        try:
            ans = minimize(self.function, self.x0, jac=self.jac)
            self.x = ans['x']
            s = f'x: {ans["x"]}\ny: {ans["fun"]}\nstatus: {ans["message"]}'
        except:
            s = f'x: {self.x0}\ny: {self.function(self.x0)}\nstatus: {"ERROR"}'
        return s


if __name__ == '__main__':
    f = 'x1**2 + exp(x2)'
    # xs = get_variables(f)
    # f = prepare_func_newton(f, xs)
    task = NewtonCG(f, [2., 2.])
    a = task.solve()
    print(a)
    print(f([0, 4]))