# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 23:19:43 2022

@author: schue
"""

import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print("R_squared:", r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

import numpy as np
from scipy import stats

x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])

slope, intercept, r, p, std_err = stats.linregress(x, y)
print("R_squared:", r**2)
print('intercept:', intercept)
print('slope:', slope)