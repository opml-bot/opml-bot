import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

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