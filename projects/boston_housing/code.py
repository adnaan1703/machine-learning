import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3


def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    regressor = DecisionTreeRegressor(random_state=0)
    params = {'max_depth': list(range(1, 11))}
    scoring_fnc = make_scorer(r2_score)
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_


data = pd.read_csv('./housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 0, shuffle = True)
reg = fit_model(X_train, y_train)

for i, price in enumerate(reg.predict(client_data)):
    print('Predicted selling price of the client {} is {:,.2f}'.format(i+1, price))
