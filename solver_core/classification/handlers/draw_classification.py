def draw(X_test, y_test, y_pred):
    import numpy as np
    if X_test.shape[1] == 2:
        import plotly.graph_objects as go
        X = np.concatenate((X_test, y_test, y_pred), axis=1)
        fig = go.Figure()
        symb = ['circle' if i == np.unique(X[:, 2])[0] else 'square' for i in X[:, 2]]
        col = ['red' if i == np.unique(X[:, 3])[0] else 'blue' for i in X[:, 3]]
        fig.add_trace(go.Scatter(x=X[:, 0],
                                 y=X[:, 1],
                                 mode='markers',
                                 marker=dict(
                                     symbol=symb,
                                     size=15,
                                     color=col)))
        return fig
    elif X_test.shape[1] == 3:
        import plotly.graph_objects as go
        X = np.concatenate((X_test, y_test, y_pred), axis=1)
        fig = go.Figure()
        symb = ['circle' if i == np.unique(X[:, 3])[0] else 'square' for i in X[:, 3]]
        col = ['red' if i == np.unique(X[:, 4])[0] else 'blue' for i in X[:, 4]]
        fig.add_trace(go.Scatter3d(x=X[:, 0],
                                   y=X[:, 1],
                                   z=X[:, 2],
                                   mode='markers',
                                   marker=dict(
                                       symbol=symb,
                                       size=10,
                                       color=col)))
        return fig
    else:
        raise ValueError('Невозможно нарисовать')
