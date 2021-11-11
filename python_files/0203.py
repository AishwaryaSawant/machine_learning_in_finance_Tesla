# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:49:18 2018
Mixture Model
Three experiments:    
1. how to construct u
2. change mu1, mu2
3. change sig1 sig2

@author: Yizhen Zhao
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; 

"Component 1: N(5,2)"
mu1 = 170
sig1 = 10

"Component 2: N(10,1)"
mu2 = 240
sig2 = 5

"Sample Size"
T = 1000

x1 = np.random.normal(mu1,sig1,T);
x2 = np.random.normal(mu2,sig2,T);
'''
y = x1+x2
'''
u = 1*(np.random.uniform(0,1,T)<=0.7)

" NOTE: 50 percent of probability to be 1"
" 50 percent of probability to be 0"
y = x1*u+(1-u)*x2
plt.subplot(2,1,1)
plt.hist(y,100)
plt.subplot(2,1,2)
sns.kdeplot(y,shade=True, color="r")
