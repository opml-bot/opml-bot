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

if __name__ == "__main__":
    # generate random exp regression
    n_samples = 100
    n_features = 1
    noise = 0
    a = 100
    b = np.array([0.2])
    # b = np.array([0.2, 0.6, 0.1])
    # a = np.random.random()*100
    # b = np.random.random(n_features)*5

    X = 50*np.random.random(n_samples * n_features).reshape((n_samples, n_features))
    Y = (a * np.exp(X @ b.reshape((-1, 1)))).flatten() + noise * np.random.random(n_samples)

    task = ExpRegression(X=X, y=Y)
    s = task.solve()
    print(s)
    #print(task.r2())
    draw_2d(task)