#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
root = '/Users/yunzhugu/MLSys-NYU-2022/weeks/2/data/'
df = pd.read_csv(root+'train.csv')
X = pd.DataFrame()
X = pd.concat((df.loc[:,("1stFlrSF","2ndFlrSF")],df.loc[:,"TotalBsmtSF"]), axis = 1)
y = df.loc[:,"SalePrice"]
beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
print("beta", beta)
y_hat = np.dot(X,beta)
R_square = 1 - ( np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2))
print("R_square", R_square)

