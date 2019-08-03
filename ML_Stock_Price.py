# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:34:03 2019

@author: Jéhoiakim KINGNE
"""

# LIBRARIES IMPORTATION
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
from functions import WriteListToCSV, extract
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV



#URL

URL = "https://www.quandl.com/api/v3/datasets/EURONEXT/NOKIA.json?api_key=F9xUFfqyGZdfFeh9stsv"

# DATA COLLECTION

quandlData = extract(URL)
columns = quandlData['dataset']['column_names']
raw_data = quandlData['dataset']['data']

currentPath = os.getcwd()
csv_file = currentPath + "/allData.csv"
WriteListToCSV(csv_file,columns,raw_data)

pd_data = pd.read_csv('allData.csv')
dateparser = lambda date: pd.datetime.strptime(date, '%Y-%m-%d') 
ts_data = pd.read_csv('allData.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparser)


# DATA CLEANING 

""" We do some statistics to vizualise ans analyze all the data """

describe = pd_data.describe()




# VISUALIZATION OF THE TRUE END OF PRICE OF NOKIA FROM 2015-11-19 TO 2019-08-02
plt.figure()
true_data = ts_data[['Last']]
plt.plot(true_data)
plt.title('NOKIA stock price evolution')



# DATA PREPARATION AND FEATURE SCALING

""" We will scale the Volume feature using standardization """

Volume = (pd_data['Volume'] - pd_data['Volume'].mean())/ pd_data['Volume'].std()

""" We add some additionnal features """
 
pd_data[['Year', 'Month', 'Day']] = pd.DataFrame([ x.split("-") for x in pd_data['Date'].tolist() ])
pd_data['scaledVolume'] = Volume


""" We will select the meaningful features with PCA to avoid overfitting """

stock_info = pd_data
stock_info = stock_info.drop('Year', axis = 1) #Not useful
stock_info = stock_info.drop('Last', axis = 1) #It is our label
stock_info = stock_info.drop('Date', axis = 1) #Not useful
stock_info = np.array(stock_info)
pca = PCA(n_components=8)
pca.fit(stock_info)
variance_ratio = pca.explained_variance_ratio_

""" The open price seems to be the most important factor determining the close price followed by the High Price. 
We will first of all consider these two features """


""" We select our features """

stock_features = (pd_data[['Open', 'High']])



""" Our label will definitely be the end of day price of NOKIA stock """


""" Labels """

stock_label = np.array(pd_data['Last'])



# TRAIN AND TEST SETS

train_stock_features, test_stock_features, train_stock_label, test_stock_label = train_test_split(stock_features, stock_label, test_size = 0.3, random_state = 0)


# TRAIN MODEL

model = RandomForestRegressor(n_estimators = 1000, random_state = 0)
model.fit(train_stock_features, train_stock_label)

#Prediction

prices_prediction = model.predict(test_stock_features)

#Error
errors = abs(prices_prediction - test_stock_label)

#Mean absolute error
print('Mean Absolute Error:', round(np.mean(errors), 2), '$') #Mean absolute error is about 0.4$


                                 
""" Now let's take more features (open, high and low prices, month and scaled volume) and see what happen """

stock_features_2 = np.array((pd_data[['Open', 'High', 'Low', 'Month', 'scaledVolume']]))

train_stock_features_2, test_stock_features_2, train_stock_label_2, test_stock_label_2 = train_test_split(stock_features_2, stock_label, test_size = 0.3, random_state = 0)
model_2 = RandomForestRegressor(n_estimators = 1000, max_features = 5, random_state = 0)
model_2.fit(train_stock_features_2, train_stock_label)
prices_prediction_2 = model_2.predict(test_stock_features_2)
errors_2 = abs(prices_prediction_2 - test_stock_label)
print('Mean Absolute Error:', round(np.mean(errors_2), 2), '$') #Mean absolute error is about 0.3$

""" We are seeing that while considering more features we are not overfitting, and the result is better """

# TUNNING THE MODEL HYPERPARAMETERS

""" The following are the parameters we'll try to evaluate to optimize our model """

#n_estimators = number of trees in the foreset
#max_features = max number of features considered for splitting a node
#max_depth = max number of levels in each decision tree
#min_samples_split = min number of data points placed in a node before the node is split
#min_samples_leaf = min number of data points allowed in a leaf node
#bootstrap = method for sampling data points (with or without replacement)

""" We will use random search to optimize our model """
#That is setting up a grid of hyperparameter values and select random combinations to train the model and score. 
#The number of search iterations is set based on time/resources. 


""" We set some values for our parameters """

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ["auto", "sqrt", "log2"]
max_depth = [int(x) for x in np.linspace(start = 10, stop = 50, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
bootstrap = [True, False]

#We create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#Random search of parameters, using 4 fold cross validation, 
#search across 50 different combinations, and use all available cores

model_random = RandomizedSearchCV(estimator = model_2, param_distributions = random_grid, n_iter = 50, cv = 4, verbose=2, random_state=0, n_jobs = -1)

model_random.fit(train_stock_features_2, train_stock_label)
print(model_random.best_params_)


prices_prediction_3 = model_random.predict(test_stock_features_2)
errors_3 = abs(prices_prediction_3 - test_stock_label)
print('Mean Absolute Error:', round(np.mean(errors_3), 2), '$') #Mean absolute error is about 0.3$


