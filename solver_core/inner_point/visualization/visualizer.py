import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


if __name__ == "__main__":
    from solver_core.inner_point.FirstPhase import FirstPhase
    from solver_core.inner_point.barriers import LogBarrirers
    from solver_core.inner_point.handlers.input_validation import *
    from solver_core.inner_point.handlers.prepocessing import prepare_all
else:
    from .FirstPhase import FirstPhase


def plot():
    pass


def plot_constraints(restr: list, started_point: np.ndarray):
    """
    Функция для отрисовки ограничений в двумерном случае.
    """
    restr_for_draw = []
    for i in restr:
        b = i(np.zeros(2))
        a1 = i(np.array([1, 0])) - b
        a2 = i(np.array([0, 1])) - b
        func_from_restr = lambda x: (-b - a2 * x) / a1  # ограничение все еще менбше нуля, так что закрашивать снизу
        restr_for_draw.append(func_from_restr)


if __name__ == '__main__':
    f = 'x1**2 + x2**2 + (0.5*1*x1 + 0.5*2*x2)**2 + (0.5*1*x1 + 0.5*2*x2)**4'
    subject_to = 'x1+x2<=0;2*x1-3*x2<=1'
    zakharov_point_min = np.array([0, 0])
    zakharov_point_start = '-5;4'

    f = check_expression(f)
    subject_to = check_restr(subject_to, method='log_barrier')
    zakharov_point_start = check_point(zakharov_point_start, f, subject_to, 'log_barrier')

    f, subject_to, zakharov_point_start = prepare_all(f, subject_to, 'log_barrier', zakharov_point_start, ds=2)

    print(zakharov_point_start)
    for i in subject_to:
        print(i(zakharov_point_start), i(zakharov_point_start) <= 0)
    task = LogBarrirers(f, subject_to, zakharov_point_start, mu=2)
    ans = task.solve()
