#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment CP365 W1D4 : Logistic Regression Machine with Regularization

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class LogisticRegression(object):
    """

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    reg : non-negative float
        Regularization term.  reg == 0  is no regularization

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, reg = 0, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.reg = reg
        self.w_ = []
        self.errors = []

    def _row_product(self, M, coeffs):
    	coeffs = np.asarray(coeffs)
    	for i in range(M.shape[0]):
    		for j in range(M.shape[1]):
    			M[i, j] *= coeffs[i]

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values. 
        IMPORTANT: Labels for Logistic Regression must be 1 or 0!

        Returns
        -------
        self : object

        """
        self.w_ = np.random.RandomState(self.random_state).uniform(-1, 1, X.shape[1]+1)
        y = np.asmatrix(y)
        for _ in range(self.n_iter):
            delta = self.activation(X)

            with_bias = np.hstack((np.ones((X.shape[0], 1)), X))

            left_term = self.activation(X)
            self._row_product(left_term, y)

            right_term = self.activation(-X)
            self._row_product(right_term, y-1.0)

            delta = (left_term + right_term) * with_bias

            self.w_ += delta
        return self

    def net_input(self, X):
        """Calculate net input"""

    def predict(self, X):
        """Return class label after unit step"""

        return np.where(self.activation(X) > 0.5, 1, 0) (with_bias * self.w_)
        
    def activation(self,X):
        with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        e_powers = with_bias*self.w_
        return 1.0 / np.add(np.exp(e_powers), 1.0)

    def plot_decision_regions(self, X, y, resolution=0.02):

	    # setup marker generator and color map
	    markers = ('s', 'x', 'o', '^', 'v')
	    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	    cmap = ListedColormap(colors[:len(np.unique(y))])

	    # plot the decision surface
	    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
	                           np.arange(x2_min, x2_max, resolution))
	    Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
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

	    plt.xlabel('sepal length [cm]')
	    plt.ylabel('petal length [cm]')
	    plt.legend(loc='upper left')
	    plt.show()



def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()
    X = df.iloc[0:150, [0, 2]].values

    y = [1]*50 + [-1]*100
    ada1 = LogisticRegression(eta=0.0001, n_iter=100, random_state=1)
    ada1.fit(X, y)

if __name__ == '__main__':
    main()
