import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def make_descent_history(history: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает преобразованный объект истории со столбцами ['x', 'f', 'итерация'] в pd.DataFrame
    со столбцами ['x', 'y', 'z', 'итерация']::

    """

    X = []
    for x in history.x:
        X.append(list(x))

    x, y = np.array(X).T
    output_data = pd.DataFrame({'x': x, 'y': y, 'z': history['f'], 'iteration': history['iteration']})
    return output_data


def make_surface(function, bounds, cnt_dots):
    assert len(bounds) == 2, 'two tuples are required'
    assert len(bounds[0]) == 2 and len(bounds[1]) == 2, 'both tuples must have 2 numbers'
    x_axis = np.linspace(bounds[0][0], bounds[0][1], cnt_dots)
    y_axis = np.linspace(bounds[1][0], bounds[1][1], cnt_dots)
    z_axis = []
    for x in x_axis:
        z_axis_i = []
        for y in y_axis:
            z_axis_i.append(function([x, y]))
        z_axis.append(z_axis_i)

    return go.Surface(x=x_axis, y=y_axis, z=np.transpose(z_axis), colorscale='ice', name='f(x, y)', opacity=0.75)


def make_ranges(descent_history):
    min_x = descent_history.x.min()
    max_x = descent_history.x.max()

    min_y = descent_history.y.min()
    max_y = descent_history.y.max()

    x_range = [min_x - (max_x - min_x) * 0.1, max_x + (max_x - min_x) * 0.1]
    y_range = [min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1]

    return x_range, y_range


def make_descent_frames_3d(function, descent_history):
    frames = []
    draw_descent = [[], [], []]

    for i in range(descent_history.shape[0]):

        if i > 0:
            x0, x1 = descent_history.x[i - 1], descent_history.x[i]
            y0, y1 = descent_history.y[i - 1], descent_history.y[i]

            for alpha in np.linspace(0, 1, 10):
                draw_descent[0].append(x0 * alpha + x1 * (1 - alpha))
                draw_descent[1].append(y0 * alpha + y1 * (1 - alpha))
                draw_descent[2].append(function([draw_descent[0][-1], draw_descent[1][-1]]))
            else:
                draw_descent[0].append(np.nan)
                draw_descent[1].append(np.nan)
                draw_descent[2].append(np.nan)

        scatter_line = go.Scatter3d(x=draw_descent[0],
                                    y=draw_descent[1],
                                    z=draw_descent[2],
                                    name='descent',
                                    mode='lines',
                                    line={'width': 4, 'color': 'rgb(1, 23, 47)'})

        scatter_points = go.Scatter3d(x=descent_history.x[:max(1, i)],
                                      y=descent_history.y[:max(1, i)],
                                      z=descent_history.z[:max(1, i)],
                                      name='descent',
                                      mode='markers',
                                      marker={'size': 5, 'color': 'rgb(1, 23, 47)'},
                                      showlegend=False)

        frames.append(go.Frame(data=[scatter_points, scatter_line], name=i, traces=[1, 2]))

    return frames


def animated_surface(function, history, cnt_dots=100):
    descent_history = make_descent_history(history)
    bounds = make_ranges(descent_history)

    first_point = go.Scatter3d(x=descent_history.x[:1],
                               y=descent_history.y[:1],
                               z=descent_history.z[:1],
                               mode='markers',
                               marker={'size': 5, 'color': 'rgb(1, 23, 47)'},
                               showlegend=False)

    surface = make_surface(function, bounds, cnt_dots=cnt_dots)
    layout = px.scatter_3d(descent_history, x='x', y='y', z='z', animation_frame='iteration').layout
    frames = make_descent_frames_3d(function, descent_history)

    fig = go.Figure(data=[surface, first_point, first_point],
                    layout=layout, frames=frames)
    fig.update_scenes(
        xaxis_title=r'<b>x</b>',
        yaxis_title=r'<b>y</b>',
        zaxis_title=r'<b>z</b>'
    )
    fig.update_layout({'title': r'<b>Surface with optimization steps</b>'})
    return fig


if __name__ == '__main__':
    from solver_core.solver_gradient.gradient_descent_const import GradientDescentConst

    func = lambda x: x[0] ** 2 + x[1] ** 2
    gradient = lambda z, x: np.array([2 * x[0], 2 * x[1]])
    point = [5, 5]
    task = GradientDescentConst(function=func, gradient=gradient, started_point=point, save_iters_df=True)
    answer = task.solve()

    animated_surface(func, task.history)
