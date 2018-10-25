#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement the Adaline algorithm with stochastic gradient descent
"""

# ## Implementing an adaptive linear neuron in Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.0001, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.cost_ = []

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.random.RandomState(self.random_state).uniform(-1, 1, X.shape[1]+1)

        for _ in range(self.n_iter):
            cur_cost = 0.0
            for i in range(X.shape[0]):
                data = np.asmatrix(X[i,:])
                label = y[i]
                with_bias = np.hstack((np.ones((1, 1)), data))
                cur_cost += np.square(self.net_input(data))
                delta = -2*(self.net_input(data) - label)*with_bias
                delta = self.eta * delta
                self.w_ += np.squeeze(np.asarray(delta))

            self.cost_.append(cur_cost[0, 0])
        return self

    def net_input(self, X):
        """Calculate net input"""
        with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.asmatrix(self.w_)*with_bias.T

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.sign(self.net_input(X))
        

def main():

    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', header=None)
    df.tail()
    X = df.iloc[0:150, [0, 2]].values
    y = [1]*50 + [-1]*100

    # Learning rate
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada1 = AdalineSGD(n_iter=10, eta=0.01).fit(X, y)
    print(ada1.cost_)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline (Stochastic) - Learning rate 0.01')

    ada2 = AdalineSGD(n_iter=10, eta=0.0001).fit(X, y)
    print(ada2.cost_)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline (Stochasrtic) - Learning rate 0.0001')

    plt.show()


if __name__ == '__main__':
    main()
