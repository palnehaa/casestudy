# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:21:10 2021

@author: neha89
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('mpg.csv')

X = dataset[['mpg','cylinders']]

y = dataset['horsepower']

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(X.iloc[0,:].values.reshape(1,-1)))