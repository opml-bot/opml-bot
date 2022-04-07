from typing import Optional, Callable
import numpy as np
import pandas as pd


class GradientDescentFrac:
    """
    Класс для решения задачи оптимизаии n мерной функции методом градиента с дроблением шага.
    Parameters
    ----------
    function: Callable
        Функция для оптимизации. Задается как питоновская функция от массива, которая возвращает сколяр.

    gradient: Callable
        Градиент. Задается как питоновская функция от массива, которая возвращает массив. Функция принимает точку
        и возвращает значение градиента в точке.

    started_point: np.ndarray
        n - мерный массив, который представляет собой координаты точки, с которой будет начинать работу алгоритм.

    alpha: Optional[flaat] = 1e-1
        Коэфициент для шага.

    delta: Optional[flaat] = 1e-1
        Коэф. для дробления шага.

    max_iteration: Optional[int] = 500
        Число максимально допустимых итераций.

    acc: Optional[float] = 10 ** -5
        Точночть для критерия остановки.

    print_midterm: Optional[bool] = False
        Флаг, надо ли выводить промежуточные результаты. Промежуточные результаты будут записаны в итоговую строку.

    save_iters_df: Optional[bool] = False
        Флаг, сохранять ли результаты в pandas.DataFrame. Этот dataframe используется для построения графика.
    """

    def __init__(self,
                 function: Callable,
                 gradient: Callable,
                 started_point: np.ndarray,
                 alpha: float = 1e-1,
                 delta: float = 1e-1,
                 max_iteration: Optional[int] = 500,
                 acc: Optional[float] = 10 ** -5,
                 print_midterm: Optional[bool] = False,
                 save_iters_df: Optional[bool] = False):

        self.function = function
        self.gradient = gradient
        self.started_point = started_point
        self.alpha = alpha
        self.delta = delta
        self.max_iteration = max_iteration
        self.acc = acc
        self.print_midterm = print_midterm
        self.save_iters_df = save_iters_df
        self.history = pd.DataFrame(columns=['x', 'f', 'iteration'])

    def solve(self):
        alpha = self.alpha
        new_x = self.started_point
        grad_k = self.gradient(self.function, new_x)
        func_k = self.function(new_x)
        ans = ''

        # Будем сохранять историю для каждой итерации. Чтобы нарисовать спуск нужно точки x и значение f
        # Для этого я создал пустой датафрейм в конструкторе и буду его заполнять
        if self.save_iters_df:
            self.history.loc[0] = [np.array(new_x), func_k, 0]

        for i in range(self.max_iteration):
            if self.print_midterm:
                ans += f'iter: {i:<4}; x:{new_x}; f(x):{func_k:.5f}\n'
            t = new_x - alpha * grad_k
            func_t = self.function(t)

            while not func_t - func_k <= - alpha * 0.1 * sum(grad_k ** 2):
                alpha = alpha * self.delta
                t = new_x - alpha * grad_k
                func_t = self.function(t)

            new_x = t
            func_k = func_t
            grad_k = self.gradient(self.function, new_x)
            x_prev = new_x
            gradient_xprev = self.gradient(self.function, x_prev)

            if self.stop_criterion(gradient_xprev):
                code = 0
                break

            if self.save_iters_df:
                self.history.loc[i + 1] = [new_x, self.function(new_x), i + 1]

        else:
            code = 1
        ans += f'\nx: {new_x}\ny: {self.function(new_x)}\ncode: {code}\niters: {i + 1}'
        return ans

    def stop_criterion(self, grad):
        """
        Метод проверяет критерий остановки. В качетсве критерия остановки используется длина градиента.
        Parameters
        ----------
        grad: np.ndarray
             Градиент в точке
        Returns
        -------
        bool
            True, если достигнута заданная точность, иначе - False
        """
        gradient_len = np.sqrt(sum([i ** 2 for i in grad]))
        if gradient_len < self.acc:
            return True
        else:
            return False


if __name__ == '__main__':
    func = lambda x: x[0] ** 2 + x[1] ** 2
    gradient = lambda z, x: np.array([2 * x[0], 2 * x[1]])
    point = [5, 5]

    task = GradientDescentFrac(function=func, gradient=gradient, started_point=point)
    answer = task.solve()
    print(answer)
