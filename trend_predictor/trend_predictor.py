from sklearn.linear_model import LinearRegression
import numpy as np


class TrendPredictor:

    def __init__(self, history_data):
        self.history_data = history_data
        self.predicted_value = None
        self.predicted_trend = None
        self.today = None
        pass

    def linear_regression(self):
        self.history_data = self.history_data.dropna()
        self.today = self.history_data.index[-1]
        n = self.history_data.shape[0]
        m = self.history_data.shape[1]
        X = self.history_data.iloc[0: n-1, 0: m-1]
        X = np.array(X)
        X = X.reshape(n - 1, m - 1)
        y = np.array(self.history_data.iloc[0: n-1, m-1]).reshape(n-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        X_pred = np.array(self.history_data.iloc[n-1, 0: m-1])
        X_pred = X_pred.reshape(1, m-1)
        self.predicted_value = model.predict(X_pred)[0][0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend =  1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0

        pass

    def lasso_regression(self):
        pass

    def ridge_regression(self):
        pass

    def elastic_regression(self):
        pass

    def regression_tree(self):
        pass

    def random_forest(self):
        pass

    def support_vector_machine(self):
        pass

    def foo(self):
        pass

    def nearest_neighbours(self):
        pass