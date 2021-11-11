# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:51:27 2019
ORDINARY LEAST SQUARE REGRESSION 
@author: Yizhen Zhao
"""
'Generate or Read  Sample for Y, X'

import numpy as np
import scipy.stats as ss
import pandas as pd
from pandas_datareader import DataReader
from datetime import datetime

MSFT = DataReader('MSFT',  'yahoo', datetime(2020,1,1), datetime(2020,8,31))
Y = np.diff(np.log(MSFT['Adj Close'].values))
T = Y.shape[0];
SPY = DataReader('SPY',  'yahoo', datetime(2020,1,1), datetime(2020,8,31))
F = np.diff(np.log(SPY['Adj Close'].values))
'Add Constant to X'
X = np.column_stack([np.ones((T,1)), F])
N = X.shape[1]

'REGRESSION STARTS:'       
'Linear Regression of Y: T x 1 on' 
'Regressors X: T x N'

invXX = np.linalg.inv(X.transpose()@X)
'OLS estimator beta: N x 1'
beta_hat = invXX@X.transpose()@Y
'Predictive value of Y_t using OLS'  
y_hat = X@beta_hat;       
'Residuals from OLS: Y - X*beta'        
residuals = Y - y_hat;            
'variance of Y_t or residuals'
sigma2 = (1/T)*(residuals.transpose()@residuals)
'standard deviation of Y_t or residuals'
sig = np.sqrt(sigma2) 
'variance-covariance matrix of beta_hat'
'N x N: on-diagnal variance(beta_j)'
'N x N: off-diagnal cov(beta_i, beta_j)'
varcov_beta_hat = (sigma2)*invXX
var_beta_hat = np.sqrt(T*np.diag(varcov_beta_hat))

'Calculate R-square'
R_square = 1 - residuals.transpose()@residuals/(T*np.var(Y))
adj_R_square = 1-(1-R_square)*(T-1)/(T-N)

'Test Each Coefficient: beta_i'
't-test stat: N x 1'
t_stat = beta_hat.transpose()/var_beta_hat
' t-test significance level: N x 1'
p_val_t = 1-ss.norm.cdf(t_stat)

'Test of Joint Significance of Model'
F_stat = beta_hat.transpose()@varcov_beta_hat@beta_hat/\
         (residuals.transpose()@residuals)
'size: (1 x N)*(N x N)*(N x 1)/((1 x T) * (T x 1)) = 1 x 1'

p_val_F = 1-ss.chi2.cdf(F_stat,T-N)

REPORT = np.column_stack([beta_hat, t_stat,p_val_t])
print('Regression Statistics')
print('------------------------\n')
print(' REGRESSION STATISTICS  \n') 
print('------------------------\n')
print('beta             t_stat            p_val\n')
print(REPORT)
print('\n Joint significance of all coefficients\n',[F_stat,p_val_F])
print('R-Square is       \n',R_square)
print('Adjusted R Square \n',adj_R_square)
print('Standard Error    \n',sig)
print('Observations      \n',T) 
print('-------------------------\n')