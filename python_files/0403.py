# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:29:18 2021
Isolation Forest
@author: yizhen 
"""
import numpy as np
import scipy.stats as ss
import pandas as pd
from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# Stock Selection 
LIST = ['AMZN','TSLA','NFLX','GS','WISH','NIO','ROKU','MS','EBAY','VIAC','NVDA']
startdate = datetime(2021,1,1)
enddate = datetime(2021,5,31)
FACTOR = ['SPY', 'QQQ']        
N = len(LIST)
PORTFOLIO = pd.DataFrame(columns = LIST) 
for n in range(0, N):
 PRICE = DataReader(LIST[n], 'yahoo', startdate,enddate)
 RETURN = np.diff(np.log(PRICE['Adj Close'].values))
 PORTFOLIO[LIST[n]] = RETURN
 
 # First Pass - TIME SERIES
 T = PORTFOLIO.shape[0]
 SPY = DataReader(FACTOR[0], 'yahoo', startdate,enddate)
 F1 = np.diff(np.log(SPY['Adj Close'].values))
 QQQ = DataReader(FACTOR[1], 'yahoo', startdate,enddate)
 F2 = np.diff(np.log(QQQ['Adj Close'].values))
 X = np.asmatrix(np.column_stack([np.ones((T,1)), F1, F2]))
 K = X.shape[1]
 
beta = np.zeros([K,N])
Y = np.zeros([T,N]) 
for n in range(0, N):
    Y[:,n] = PORTFOLIO[LIST[n]].values
    #Linear Regression of Y: T x 1 on 
    # Regressors X: T x N
    invXX = np.linalg.inv(X.transpose()@X)
    #OLS estimator beta: N x 1'
    beta[:,n] = invXX@X.transpose()@Y[:,n]


fig = plt.figure(figsize = (10, 7))    
plt.scatter(beta[0,:], beta[1,:], color = 'b', s = 60)
plt.xlabel('beta to SPY')
plt.ylabel('beta to QQQ')
plt.scatter(beta[0,:][beta[1,:]>1], beta[1,:][beta[1,:]>1], color = 'r', s= 100)
plt.show()

model=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12), \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)

model.fit(beta.transpose())
scores = model.decision_function(beta.transpose())
pred = model.predict(beta.transpose())
'''
To show the predicted anomalies present in the dataset under the egg weight column, 
data need to be analyzed after the addition of scores and anomaly columns. 
Note that the anomaly column values would be -1 and 
the corresponding scores will be negative.         
'''          




