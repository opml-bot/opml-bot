from typing import Callable, Optional
import numpy as np
import pulp as plp

from solver_core.ILP.handlers.input_validation import *
from solver_core.ILP.handlers.prepocessing import prepare_all

class BNB:

    def __init__(self,
                function: Callable,
                restrictions):
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

        ansx, f_x = max(self.done, key=lambda x: x[1])
        ansx = np.array(ansx)
        return ansx, f_x

    def subproblem_solver(self, restrs):
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

    f = '3*x1+5*x2'
    subject_to = '5*x1+2*x2<=14; 2*x1+5*x2<=16;x1>=0; x2>=0'

    f = check_expression(f)
    subject_to = check_restr(subject_to, method='bnb')
    # preprocessing
    f, subject_to = prepare_all(f, subject_to, 'bnb')
    p = BNB(f, subject_to)
    x = p.solve()
    print(x)