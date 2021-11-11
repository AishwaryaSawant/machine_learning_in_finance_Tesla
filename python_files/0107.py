# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 21:28:23 2018
Multivariate Normal
@author: Yizhen Zhao
"""
import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()