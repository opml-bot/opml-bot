import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from typing import Callable, Optional

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
    fig = make_subplots(subplot_titles=['График уравнения регрессии и точек.'])
    # отрисовка точек
    x_scatter = reg_object.X.flatten()
    y_scatter = reg_object.y
    fig.add_trace(go.Scatter(x=x_scatter, y=y_scatter, mode='markers', name='Исходные точки'))
    fig.update_traces(marker=dict(size=10,
                                  color='#ff5733',
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    # отрисовка линии
    razm = x_scatter.max() - x_scatter.min()
    x_line = np.linspace(x_scatter.min() - razm*0.1, x_scatter.max() + razm*0.1, 1000)
    y_line = reg_object.predict(x_line)
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', line=dict(width=3, color='#b30fb2'), name='Линия регрессии'))
    fig.show()


def prepare_surface(X: np.ndarray, function: Callable, ndots: Optional[int] = 50):
    """
    Функция подготавливает датафрейм для отрисовки поверхности.

    Parameters
    ----------
    X: np.ndarray
        Массив точек из модели регрессии. Нужен для того чтоб определить границы отрисовки.

    function: Callable
        Метод predict из класса задачи регрессии.

    ndots: Optional[int] = 50
        Количество точек на одну ось.

    Returns
    -------
    dots: pd.DataFrame
        Датафрейм для отрисовки, где индексы - значения по оси x, имена колонок - значения по оси y, а сами значения
        и есть значения функции в точке.
    """

    razm_x1 = X[:, 0].max() - X[:, 0].min()
    razm_x2 = X[:, 1].max() - X[:, 1].min()
    x1 = np.linspace(X[:, 0].min() - 0.1 * razm_x1, X[:, 0].max() + 0.1 * razm_x1, ndots)
    x2 = np.linspace(X[:, 1].min() - 0.1 * razm_x2, X[:, 1].max() + 0.1 * razm_x2, ndots)

    dots = pd.DataFrame(index=x1, columns=x2)
    for x in range(len(x1)):
        for y in range(len(x2)):
            dots.iloc[x, y] = function(np.array([x, y]).reshape((1, 2)))[0]
    return dots


def draw_3d(reg_object: object):
    """
    Отрисовывает регрессию на трехмерном пространстве.

    Parameters
    ----------
    reg_object: object
        Объект одного из типов регресии. Должен содержать метод predict и атрибут X.

    Returns
    -------

    """
    fig = make_subplots(subplot_titles=['График уравнения регрессии и точек.'])
    # отрисовка точек
    x_scatter = reg_object.X[:, 0]
    y_scatter = reg_object.X[:, 1]
    z_scatter = reg_object.y
    fig.add_trace(go.Scatter3d(x=x_scatter, y=y_scatter, z=z_scatter, mode='markers', name='Исходные точки'))
    fig.update_traces(marker=dict(size=6,
                                  color='#ff5733',
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    # отрисовка линии
    data = prepare_surface(reg_object.X, reg_object.predict)
    fig.add_surface(x=data.index, y=data.columns, z=data.values.T, colorscale='ice', name='f(x, y)', opacity=0.75)
    fig.show()


def draw(regression: object):
    """
    Функция обертка для отрисовки. Внутри себя выполняет необходимые действия для запуска одной из функций draw_3d или
    draw_2d.
    """
    if type(regression) == ExpRegression:
        if regression.X.shape[1] == 1:
            draw_2d(regression)
        elif regression.X.shape[1] == 2:
            draw_3d(regression)
        else:
            mes = 'К сожалению, не получится построить график, так как регрессия является бооее чем трехмерной'
            raise ValueError(mes)


if __name__ == "__main__":
    # generate random exp regression
    n_samples = 100
    n_features = 1
    noise = 100
    a = 100
    b = np.array([0.02])
    # b = np.array([0.2, 0.6, 0.1])
    # a = np.random.random()*100
    # b = np.random.random(n_features)*5

    X = 50*np.random.random(n_samples * n_features).reshape((n_samples, n_features))
    Y = (a * np.exp(X @ b.reshape((-1, 1)))).flatten() + noise * np.random.random(n_samples)

    task = ExpRegression(X=X, y=Y)
    s = task.solve()
    print(s)
    print(task.r2())
    draw(task)