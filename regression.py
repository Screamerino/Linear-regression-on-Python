import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.intercept = None
        self.coefficient = None
        self.result = None

    def fit(self, X, y):
        if self.fit_intercept:
            size = X.shape[0]
            X = np.hstack((np.ones(size)[:, np.newaxis], X))
        self.result = np.linalg.inv((X.T @ X)) @ X.T @ y
        if self.fit_intercept:
            self.intercept = self.result[0, 0]
            self.coefficient = self.result[1:, 0]
        else:
            self.coefficient = self.result

    def predict(self, X):
        if self.fit_intercept:
            size = X.shape[0]
            X = np.hstack((np.ones(size)[:, np.newaxis], X))
        return np.dot(X, self.result)

    def r2_score(self, y, yhat):
        return 1 - np.sum(np.power(y - yhat, 2)) / np.sum(np.power(y - np.mean(yhat), 2))

    def rmse(self, y, yhat):
        return np.sqrt(np.sum(np.power(y - yhat, 2)) / len(y))


df = pd.read_csv("data_stage4.csv")
*x, y = df.to_numpy().T
model = CustomLinearRegression(fit_intercept=True)
regSci = LinearRegression(fit_intercept=True)
x = np.array(x).T
y = np.array(y, ndmin=2).T
model.fit(x, y)
regSci.fit(x, y)
my_yhat = model.predict(x)
yhat = regSci.predict(x)
my_yhat.shape = (my_yhat.shape[0])
yhat.shape = (yhat.shape[0])
y.shape = (y.shape[0])
my_r2 = model.r2_score(y, my_yhat)
my_rmse = model.rmse(y, my_yhat)
r2 = r2_score(y, yhat)
rmse = np.sqrt(mean_squared_error(y, yhat))

print({
        'Intercept': regSci.intercept_ - model.intercept,
        'Coefficient': regSci.coef_ - model.coefficient,
        'R2': my_r2 - r2,
        'RMSE': rmse - my_rmse
})
