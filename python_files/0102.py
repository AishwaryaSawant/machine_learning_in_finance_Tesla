# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 20:10:48 2018

@author: Yizhen Zhao

Log-normal

"""
import numpy as np
import matplotlib.pyplot as plt

mu = 2
sigma = 3
s = np.random.lognormal(mu, sigma, 1000)
print(s)
x = np.linspace(0, 400, 1000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2*sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show();
