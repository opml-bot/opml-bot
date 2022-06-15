from solver_core.inner_point.primal_dual import PrimalDual
from typing import Callable, Optional
import numpy as np
import pulp as plp

class BNB:

    def __init__(self,
                function: Callable,
                restrictions
                ):
        self.f = function
        self.restr = restrictions

    def initial_point(self):
        problem = plp.LpProblem('initial', plp.LpMaximize)
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
        xs = [plp.LpVariable(f'x{i+1}', None, None) for i in range(n_vars)]

        problem += sum(xs)
        for i in self.restr:
            problem += i(xs) >= 0
        problem.solve()
        xs = [v.varValue for v in problem.variables()]
        for i in self.restr:
            print(i(xs))


    def solve(self):
        pass

if __name__ == "__main__":
    f = lambda x: x[0] + x[1]
    consts = [lambda x: -5*x[0] + -9*x[1] - 45, lambda x: x[0], lambda x: x[1], lambda x: -x[0] - x[1] - 6]
    p = BNB(f, consts)
    p.initial_point()
