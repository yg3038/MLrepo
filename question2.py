#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
root = '/Users/yunzhugu/MLSys-NYU-2022/weeks/2/data/'
df = pd.read_csv(root+'train.csv')
X = df[['1stFlrSF','2ndFlrSF','TotalBsmtSF','LotArea','OverallQual','GrLivArea','GarageCars','GarageArea']]
y = df.loc[:,"SalePrice"]
model = LinearRegression(fit_intercept=True)
Performance = pd.DataFrame(columns = ['R2','MSE','MAE','MAPE'])
def plot_data(model,X,y):
    for i in range(1,9):
        x = X.iloc[:,:i]
        model = model.fit(x,y)
        predictions = model.predict(x)
        R2 = r2_score(y,predictions)
        MSE = mean_squared_error(y,predictions)
        MAE = mean_absolute_error(y,predictions)
        MAPE = mean_absolute_percentage_error(y,predictions)
        Performance.loc[i-1] = [R2, MSE, MAE, MAPE]
    return Performance
plot_data(model,X,y)

def plot_performance(Performance):
    for i in range(0,4):
        fig, ax = plt.subplots()
        ax.scatter(
            x = np.arange(1,9), 
            y = Performance.iloc[:,i], 
            alpha=0.25, 
        )
        ax.set_ylabel(Performance.columns[i])
        ax.set_xlabel("Variable_numbers")
        plt.show()
    return ax
plot_performance(Performance)

