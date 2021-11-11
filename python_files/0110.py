# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:02:16 2019
@author: Yizhen Zhao
Mixture Modeling: Preiliminary
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns; sns.set(color_codes=True)

'''Normal Distribution '''
u = np.random.normal(0,1,1);
mu_1 = 1200;
Sigma1 = 22500; 
r1 = mu_1 + math.sqrt(Sigma1)*u;
''' Visualize y_1 = f(x_1) '''
sigma_1 = math.sqrt(Sigma1);
x_1 = np.linspace(500, 2500, 2000)
y_1 = ss.norm.pdf(x_1, mu_1, sigma_1)

plt.figure(1)
plt.plot(x_1, y_1)

mu_2 = 1800;
Sigma2 = 22500; 
r2 = mu_2 + math.sqrt(Sigma2)*u;
''' Visualize y_2 = f(x_2) '''
sigma_2 = math.sqrt(Sigma2);
x_2 = np.linspace(500, 3000, 2500)
y_2 = ss.norm.pdf(x_2, mu_2, sigma_2)
plt.figure(2)
plt.plot(x_2, y_2)

'''Normal Mixture'''
p = 0.3; 
S = 5000;
r = np.zeros(S);
y = np.zeros(S);
for s in range(1,S):
    eps = np.random.normal(0,1,1);
    r1 = mu_1 + math.sqrt(Sigma1)*eps;
    r2 = mu_2 + math.sqrt(Sigma2)*eps;
    u = np.random.uniform(0,1,1);
    r[s] = r1*(u<p)+r2*(u>=p);

plt.figure(3)
sns.kdeplot(r,shade=True, color="r");