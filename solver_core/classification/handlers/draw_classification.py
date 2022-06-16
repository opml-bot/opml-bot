def draw(X_train, y_train):
    if X_train.shape[1] == 2:
        import plotly.graph_objects as go
        X_train_1 = X_train[np.argwhere(y_train == 1)[:, 0]]
        X_train_2 = X_train[np.argwhere(y_train == 2)[:, 0]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_train_1[:, 0],
                                 y=X_train_1[:, 1],
                                 mode='markers',
                                 marker_color='red',
                                 name='first'))
        fig.add_trace(go.Scatter(x=X_train_2[:, 0],
                                 y=X_train_2[:, 1],
                                 mode='markers',
                                 marker_color='blue',
                                 name='second'))
        fig.show()
    elif X_train.shape[1] == 3:
        import plotly.graph_objects as go
        X_train_1 = X_train[np.argwhere(y_train == 1)[:, 0]]
        X_train_2 = X_train[np.argwhere(y_train == 2)[:, 0]]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=X_train_1[:, 0],
                                   y=X_train_1[:, 1],
                                   z=X_train_1[:, 2],
                                   mode='markers',
                                   marker_color='red',
                                   name='first'))
        fig.add_trace(go.Scatter3d(x=X_train_2[:, 0],
                                   y=X_train_2[:, 1],
                                   z=X_train_2[:, 2],
                                   mode='markers',
                                   marker_color='blue',
                                   name='second'))
        fig.show()
    else:
        raise ValueError('Невозможно нарисовать')