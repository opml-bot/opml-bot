def draw(X_test, y_test, y_pred):
    import numpy as np
    if X_test.shape[1] == 2:
        import plotly.graph_objects as go
        X = np.concatenate((X_test, y_test, y_pred), axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[(X[:, 2] == np.unique(X[:, 2])[0]) & (X[:, 3] == np.unique(X[:, 3])[0])][:, 0],
                                 y=X[(X[:, 2] == np.unique(X[:, 2])[0]) & (X[:, 3] == np.unique(X[:, 3])[0])][:, 1],
                                 mode='markers',
                                 name='test 0 predict 0',
                                 marker=dict(
                                     symbol='circle',
                                     size=10,
                                     color='red')))
        try:
            fig.add_trace(go.Scatter(x=X[(X[:, 2] == np.unique(X[:, 2])[0]) & (X[:, 3] == np.unique(X[:, 3])[1])][:, 0],
                                     y=X[(X[:, 2] == np.unique(X[:, 2])[0]) & (X[:, 3] == np.unique(X[:, 3])[1])][:, 1],
                                     mode='markers',
                                     name='test 0 predict 1',
                                     marker=dict(
                                         symbol='circle',
                                         size=10,
                                         color='blue')))
        except:
            pass
        try:
            fig.add_trace(go.Scatter(x=X[(X[:, 2] == np.unique(X[:, 2])[1]) & (X[:, 3] == np.unique(X[:, 3])[0])][:, 0],
                                     y=X[(X[:, 2] == np.unique(X[:, 2])[1]) & (X[:, 3] == np.unique(X[:, 3])[0])][:, 1],
                                     mode='markers',
                                     name='test 1 predict 0',
                                     marker=dict(
                                         symbol='square',
                                         size=10,
                                         color='red')))
        except:
            pass
        try:
            fig.add_trace(go.Scatter(x=X[(X[:, 2] == np.unique(X[:, 2])[1]) & (X[:, 3] == np.unique(X[:, 3])[1])][:, 0],
                                     y=X[(X[:, 2] == np.unique(X[:, 2])[1]) & (X[:, 3] == np.unique(X[:, 3])[1])][:, 1],
                                     mode='markers',
                                     name='test 1 predict 1',
                                     marker=dict(
                                         symbol='square',
                                         size=10,
                                         color='blue')))
        except:
            pass
        fig.update_layout(title="Результаты классификации на тестовой выборке", xaxis_title="X, у.е.",
                          yaxis_title="Y, у.е.")
        fig.update_traces(showlegend=True)
        return fig
    elif X_test.shape[1] == 3:
        import plotly.graph_objects as go
        X = np.concatenate((X_test, y_test, y_pred), axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=X[(X[:, 3] == np.unique(X[:, 3])[0]) & (X[:, 4] == np.unique(X[:, 4])[0])][:, 0],
                                   y=X[(X[:, 3] == np.unique(X[:, 3])[0]) & (X[:, 4] == np.unique(X[:, 4])[0])][:, 1],
                                   z=X[(X[:, 3] == np.unique(X[:, 3])[0]) & (X[:, 4] == np.unique(X[:, 4])[0])][:, 2],
                                   mode='markers',
                                   name='test 0 predict 0',
                                   marker=dict(
                                       symbol='circle',
                                       size=10,
                                       color='red')))
        try:
            fig.add_trace(go.Scatter3d(x=X[(X[:, 3] == np.unique(X[:, 3])[0]) & (X[:, 4] == np.unique(X[:, 4])[1])][:, 0],
                                   y=X[(X[:, 3] == np.unique(X[:, 3])[0]) & (X[:, 4] == np.unique(X[:, 4])[1])][:, 1],
                                   z=X[(X[:, 3] == np.unique(X[:, 3])[0]) & (X[:, 4] == np.unique(X[:, 4])[1])][:, 2],
                                   mode='markers',
                                   name='test 0 predict 1',
                                   marker=dict(
                                       symbol='circle',
                                       size=10,
                                       color='blue')))
        except:
            pass
        try:
            fig.add_trace(go.Scatter3d(x=X[(X[:, 3] == np.unique(X[:, 3])[1]) & (X[:, 4] == np.unique(X[:, 4])[0])][:, 0],
                                       y=X[(X[:, 3] == np.unique(X[:, 3])[1]) & (X[:, 4] == np.unique(X[:, 4])[0])][:, 1],
                                       z=X[(X[:, 3] == np.unique(X[:, 3])[1]) & (X[:, 4] == np.unique(X[:, 4])[0])][:, 2],
                                       mode='markers',
                                       name='test 1 predict 0',
                                       marker=dict(
                                           symbol='square',
                                           size=10,
                                           color='red')))
        except:
            pass
        try:
            fig.add_trace(go.Scatter3d(x=X[(X[:, 3] == np.unique(X[:, 3])[1]) & (X[:, 4] == np.unique(X[:, 4])[1])][:, 0],
                                       y=X[(X[:, 3] == np.unique(X[:, 3])[1]) & (X[:, 4] == np.unique(X[:, 4])[1])][:, 1],
                                       z=X[(X[:, 3] == np.unique(X[:, 3])[1]) & (X[:, 4] == np.unique(X[:, 4])[1])][:, 2],
                                       mode='markers',
                                       name='test 1 predict 1',
                                       marker=dict(
                                           symbol='square',
                                           size=10,
                                           color='blue')))
        except:
            pass
        fig.update_layout(title="Результаты классификации на тестовой выборке", xaxis_title="X, у.е.",
                          yaxis_title="Y, у.е.")
        fig.update_traces(showlegend=True)

        return fig
    else:
        raise ValueError('Невозможно нарисовать')
