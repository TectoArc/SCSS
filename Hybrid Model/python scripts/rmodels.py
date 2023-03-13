import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.pipeline import make_pipeline

class RModels:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
    
        self.lin = LinearRegression()
        self.ridge = Ridge(alpha=1.0)
        self.pls = PLSRegression(n_components=2)
        self.knn = KNeighborsRegressor(n_neighbors=5)
        self.br = linear_model.BayesianRidge()
        self.poly = make_pipeline(PolynomialFeatures(degree=2), self.lin)

    def optimize_rm(self):
        """
        Ridge Regression optimize : parameter - alpha 
        """
        r_cv = RidgeCV()
        r_cv.fit(self.x_train, self.y_train)
        self.ridge = Ridge(r_cv.alpha_)
        """
        KNearest Neighbors Regressor optimize : parameter - n_neighbors
        """
        rmse_knn, err_knn = [], []
        for k in range(2,20):
            knn = KNeighborsRegressor(n_neighbors=k)
            pred = knn.fit(self.x_train, self.y_train).predict(self.x_test)
            err_knn.append(np.sqrt(mean_squared_error(self.y_test, pred)))
            rmse_knn.append([k, np.sqrt(mean_squared_error(self.y_test, pred))])

        n = [x for (x, y) in rmse_knn if y==min(err_knn)]
        if len(n) != 0:
            self.knn = KNeighborsRegressor(n_neighbors=n.pop())
        else:
            self.knn = self.knn

    def predict_rm(self, df, optimize=True):
        if optimize:
            self.optimize_rm()
        y_pred_df = []
        estimators = [self.lin, self.ridge, self.poly, self.pls, self.knn, self.br]
        for i, estimator in enumerate(estimators):
            y_pred_df.append([estimator.fit(self.x_train, self.y_train).predict(df)])
        return y_pred_df


if __name__ == "__main__":
    print('RModels called successfully')