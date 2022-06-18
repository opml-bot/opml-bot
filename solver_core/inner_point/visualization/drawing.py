import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


if __name__ == "__main__":
    from solver_core.inner_point.FirstPhase import FirstPhase
else:
    from .FirstPhase import FirstPhase

def plot():
    pass


def plot_constraints(restr: list):
    """
    Функция для отрисовки ограничений в двумерном случае.
    """
    restr_for_draw = []
    for i in restr:
        b = i(np.zeros(2))
        a1 = i(np.array([1, 0])) - b
        a2 = i(np.array([0, 1])) - b
        func_from_restr = lambda x: (-b - a2*x)/a1 # ограничение все еще менбше нуля, так что закрашивать снизу
        restr_for_draw.append(func_from_restr)


def
