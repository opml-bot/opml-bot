import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from solver_core.regression.linear_regression import LinearRegression
from solver_core.regression.exponential_regression import  ExpRegression


def draw_2d(reg_object: object):
    """
    Отрисовывает регрессию на плоскости.

    Parameters
    ----------
    reg_object: object
        Объект одного из типов регресии. Должен содержать метод predict и атрибут X.

    Returns
    -------

    """
    fig = make_subplots()
    xs = reg_object.X.flatten()
    fig.add_trace(reg_object.X.flatten)

if __name__ == "__main__":
    X = (np.random.random(100)*100).reshape((-1, 1))
    X = np.random.random(100)*100
    y = 20 + X[:, 0]*2 + X[:, 0]*3
    task = LinearRegression(X=X, y=y)
    task.solve()