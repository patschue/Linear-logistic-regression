# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 23:19:43 2022

@author: schue
"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# Import File and overview
df = pd.read_csv("employees.csv", sep=",")
df.head()

# First print data
df.plot.scatter(x = "Age", y = "AbsentHours")
plt.show()

# Data wrangling
df = df[df["AbsentHours"] != 0]
df = df.dropna()
df.plot.scatter(x = "Age", y = "AbsentHours")
plt.show()

# Prepare fit
age = df["Age"].values.reshape(-1,1)
absenthours = df["AbsentHours"].values.reshape(-1,1)

# Model fit
model = LinearRegression().fit(age, absenthours)
print("r_sq", model.score(age,absenthours))
print("intercept", model.intercept_)
print("slope", model.coef_)

# Predict for chart
x_new = np.arange(60).reshape(-1,1)+ 20
y_new = model.predict(x_new)

df.plot.scatter(x = "Age", y = "AbsentHours")
plt.plot(x_new, y_new, color = "r")
plt.show()

# Predict residuals
predictedabsence = model.predict(age)
resid = absenthours - predictedabsence

# Scatterplot residuals
plt.plot(age, resid, "o")
plt.show()

# Histogram residuals
n, bins, patches = plt.hist(resid, bins=150, facecolor='blue',stacked=True,density=True)
plt.xlabel('Residuen')
plt.ylabel('Haeufigkeit')

mu = np.average(resid)
sigma = np.std(resid)

y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')