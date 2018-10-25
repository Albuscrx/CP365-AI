#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:38:43 2018

@author: richard
"""

#Class Code  W1D2

#Useful imports
#numpy for linear algebra calculations
import numpy as np
#pandas for data frames and IO
import pandas as pd
#plotting tools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Perceptron import Perceptron


#Load Iris dataset from UCI repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
#Tail gives the last few rows of the dataframe
df.tail()
# extract sepal length and petal length from first 100 data vectors
X = df.iloc[0:100, [0, 2]].values
#Plot labeled data. Note the first 50 instances are of class setosa
#and the second 50 instances are of class versicolor
# plot data

#assuming your Perceptron class computes the number of errors in each Epoch
#and stores the values in an array self.errors_, the following code will
#create a plot of the errors as a function of number of Epochs

import random
y = [1]*50 + [-1]*50
ppn = Perceptron(eta=0.1, n_iter=10, random_state=random.randint(1,65536))

ppn.fit(X, y)

'''
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.show()
'''
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

plot_decision_regions(X, y, ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


# plt.savefig('images/02_08.png', dpi=300)
plt.show()
