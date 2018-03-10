import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge

class TrendPredictor:

    def __init__(self, history_data):
        self.history_data = history_data

        self.predictors = None
        self.response = None
        self.n = None
        self.m = None

        self.predicted_value = None
        self.predicted_trend = None
        self.today = None
        pass

    def prepare_data(self):
        self.history_data = self.history_data.dropna()
        self.today = self.history_data.index[-1]
        self.n = self.history_data.shape[0]
        self.m = self.history_data.shape[1]
        X = self.history_data.iloc[0: self.n - 1, 0: self.m - 1]
        X = np.array(X)
        self.predictors = X.reshape(self.n - 1, self.m - 1)
        self.response = np.array(self.history_data.iloc[0: self.n - 1, self.m - 1]).reshape(self.n - 1, 1)


    def linear_regression(self):

        self.prepare_data()

        X = self.predictors
        y = self.response

        model = LinearRegression()
        model.fit(X, y)

        X_pred = np.array(self.history_data.iloc[self.n-1, 0: self.m-1])
        X_pred = X_pred.reshape(1, self.m-1)
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
        self.prepare_data()
        X = self.predictors
        y = self.response

        model = Lasso(alpha = 0.1)
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0

        pass

    def ridge_regression(self):
        self.prepare_data()
        X = self.predictors
        y = self.response

        model = Ridge(alpha=2, solver='cholesky')
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0][0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0

        pass

    def elastic_regression(self):

        self.prepare_data()
        X = self.predictors
        y = self.response
        model = ElasticNet(alpha=2, l1_ratio=0.5)
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0
        pass

    def pls_regression(self):
        self.prepare_data()
        X = self.predictors
        y = self.response
        model = PLSRegression(n_components=2, scale=False)
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0][0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0

    def regression_tree(self):
        self.prepare_data()
        X = self.predictors
        y = self.response

        model = DecisionTreeRegressor()
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0
        pass

    def random_forest(self):
        self.prepare_data()
        X = self.predictors
        y = self.response
        y = np.array(y).reshape(-1)
        model = RandomForestRegressor()
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0
        pass

    def support_vector_machine(self):
        self.prepare_data()
        X = self.predictors
        y = self.response
        y = y.reshape(-1)

        model = SVR(C=1.0, epsilon=0.2, kernel='rbf')
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0
        pass

    def nearest_neighbours(self):
        self.prepare_data()
        X = self.predictors
        y = self.response

        model = KNeighborsRegressor()
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0][0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0
        pass

    def bayesian_regression(self):
        self.prepare_data()
        X = self.predictors
        y = self.response

        model = BayesianRidge()
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0][0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0

    def neural_net(self):
        pass