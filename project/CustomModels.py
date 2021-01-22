from collections import defaultdict

import numpy as np
import numpy.linalg as la
from sklearn.linear_model import Ridge

def generate_ar_frame(data, lag_order):
    A_windows = (
        np.expand_dims(np.arange(lag_order), 0) +
        np.expand_dims(np.arange(data.shape[0] - lag_order), 0).T
    )
    b_indices = np.arange(lag_order, data.shape[0])

    A = data[A_windows]
    A = A.reshape((A.shape[0], A.shape[1] * A.shape[2]))
    b = data[b_indices]

    return A, b

class VAR:
    def __init__(self, lag_order):
        self.lag_order = lag_order

    def fit(self, data, data_exog=None):
        assert data_exog is None or data.shape == data_exog.shape

        A, b = generate_ar_frame(data, self.lag_order)

        A = np.nan_to_num(A)
        b = np.nan_to_num(b)

        self.x, residuals, _, _ = la.lstsq(A, b)
        return residuals

    def predict(self, X):
        assert len(X.shape) == 3

        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

        return X @ self.x



class RidgeAR:
    def __init__(self, lag_order):
        self.lag_order = lag_order
        self.regressor = Ridge(alpha=1)
        self.exog_ars = []
        self.n_exog = -1

    def fit(self, data):
        self.regressor.fit(data[:, 1:], data[:, 0])
        self.n_exog = data.shape[1] - 1
        self.exog_ars = [Ridge() for _ in range(self.n_exog)]
        
        for i in range(1, data.shape[1]):
            X, y = generate_ar_frame(data[:, i, None], self.lag_order)
            self.exog_ars[i - 1].fit(X, y)

    def predict(self, lag_data):
        assert lag_data.shape == (self.lag_order, self.n_exog + 1)

        exog_pred = np.empty(self.n_exog)
        for i, exog_ar in enumerate(self.exog_ars):
            exog_pred[i] = exog_ar.predict(lag_data[None, :, i + 1])[0]
        
        reg_pred = self.regressor.predict(exog_pred[None, :])[0]

        return np.hstack((reg_pred, exog_pred))

class AR:
    def __init__(self, lag_order):
        self.lag_order = lag_order
        self.regressors = []
        self.n_regs = -1

    def fit(self, data):
        self.n_regs = data.shape[1]

        for i in range(self.n_regs):
            model = Ridge()
            X, y = generate_ar_frame(data[:, i, None], self.lag_order)
            model.fit(X, y)
            self.regressors.append(model)
    
    def predict(self, lag_data):
        assert lag_data.shape == (self.lag_order, self.n_regs)

        pred = []
        for i, reg in enumerate(self.regressors):
            pred.append(reg.predict(lag_data[None, :, i])[0, 0])
        
        return np.array(pred)