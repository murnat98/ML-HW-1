import numpy as np
import pandas as pd
from lin_reg import LinearRegression

def fit_methods(methods, x, y):
    for method in methods:
        model = LinearRegression(method=method)
        weights = model.fit(x, y)

        print(model.loss_function())
        print(weights)

df = pd.read_csv('ecommerce.csv')
df = df.iloc[:, 3:]
x = df['Length of Membership'].to_numpy()
y = df['Yearly Amount Spent'].to_numpy()

fit_methods(('analytic', 'gd', 'sgd'), x, y)

# TODO: compare your results with the same models on sklearn
# LinearRegression, SGDRegressor - sklearn objects

