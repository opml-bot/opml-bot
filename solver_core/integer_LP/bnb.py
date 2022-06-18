from typing import Callable, Optional
import numpy as np
import pulp as plp

class BNB:
    """
    Класс для решения задачи целочисленного линейного программирования методом ветвей и границ.

    Parameters
    ----------
    function: Callable
        Функция для минимизации.

    restrictions: list
        Список функций (в смысле питоновских функций), которые представляют собой ограничения типа '<='.

    """
    def __init__(self,
                function: Callable,
                restrictions: list):
        self.f = function
        self.restr = restrictions
        i = 0
        while True:
            i += 1
            try:
                self.f(np.zeros(i))
            except IndexError:
                pass
            else:
                n_vars = i
                break
        self.xs = [plp.LpVariable(f'x{i+1}', None, None) for i in range(n_vars)]
        self.init_restr = [i(self.xs) <= 0 for i in self.restr]

        self.stack = [self.init_restr]
        self.solved = list()
        self.done = set()

    def solve(self):
        """
        Метод решает задачу.

        Returns
        -------
        ansx: np.ndarray
            Ответ в виде координаты точки.
        f_x: float
            Значение целевой функции в этой точке.
        """
        while self.stack:
            task = self.stack.pop()
            str_task = [f'{i}' for i in task]
            str_solved = [[f'{i}' for i in j] for j in self.solved]
            if str_task in str_solved:
                ans = False
                for i in str_solved:
                    print(i)
            else:
                ans = self.subproblem_solver(task)

            if ans:
                for i, xi in enumerate(ans):
                    if xi % 1 != 0:
                        sep_x = self.xs[i]
                        sep_val = xi
                        break
                else:
                    self.done.add((tuple(ans), self.f(ans)))
                    sep_val = False

                if sep_val:
                    self.stack.append(task + [sep_x <= sep_val//1])
                    self.stack.append(task + [sep_x >= sep_val // 1 + 1])
            self.solved.append(task)
        try:
            ansx, f_x = max(self.done, key=lambda x: x[1])
            ansx = np.array(ansx)
        except:
            ansx = 'Решение не найдено'
            f_x = None
        return ansx, f_x

    def subproblem_solver(self, restrs):
        """
        Метод решает задачу целочисленного программирования. Так как метод ветвей и границ предпологает решение
        большого количества подзадач, то возникла необходимость в написании этого метода.

        Parameters
        ----------
        restrs: list
            Массив в ограничениями для задачи. Для каждой подзадачи целевая функция и переменные одни и те же, но
            разные ограничения.

        Returns
        -------
        answer: list
            Массив с координатами точек, если решения существует. Иначе возвращается False.
        """
        problem = plp.LpProblem('initial', plp.LpMaximize)
        problem += self.f(self.xs)
        for i in restrs:
            problem += i

        solver = plp.PULP_CBC_CMD(msg=0)
        status = problem.solve(solver)
        if status == 1:
            answer = [v.varValue for v in problem.variables()]
        else:
            answer = False
        return answer


if __name__ == "__main__":
    from solver_core.integer_LP.handlers.input_validation import *
    from solver_core.integer_LP.handlers.prepocessing import prepare_all
    f = '3*x1+5*x2'
    subject_to = '5*x1+2*x2<=14; 2*x1+5*x2<=16;x1>=0; x2>=0'

    f = check_expression(f)
    subject_to = check_restr(subject_to, method='bnb')
    # preprocessing
    f, subject_to = prepare_all(f, subject_to, 'bnb')
    p = BNB(f, subject_to)
    x = p.solve()
    print(x)