# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:07:42 2021

@author: yizhe
"""

TSLA = DataReader('TSLA',  'yahoo', datetime(2020,1,1), datetime(2020,8,31))
Y = TSLA['Adj Close'].values 
T = Y.shape[0];

for t in range(1,T):
      # RECURSIVE
      y = Y[1:t]
      # BOOTSTRAP
      Y_boot = y[np.random.choice(T,T)]
    return Y_boot

k = 60;
for t in range(k+1,T):
      # ROLLING WINDOW
       y = Y[:t-k]
       Y_boot = y[np.random.choice(T,T)]
    return Y_boot