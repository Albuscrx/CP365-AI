#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement the Adaline algorithm with gradient descent
"""

# ## Implementing an adaptive linear neuron in Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
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
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
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

        with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        for _ in range(self.n_iter):
            prediction = self.net_input(X)
            self.cost_.append(np.sum(np.square(prediction-y)))
            delta = -(np.asmatrix(prediction-y))*with_bias
            delta = self.eta * delta
            self.w_ += np.squeeze(np.asarray(delta))
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
    ada1 = AdalineGD(eta=0.0001, n_iter=100, random_state=1)
    ada1.fit(X, y)
    plt.title('Adaline - Setosa classifier')
    ada1.plot_decision_regions(X, y)

    y = [-1]*50 + [1]*50 + [-1]*50
    ada2 = AdalineGD(eta=0.0001, n_iter=100, random_state=1)
    ada2.fit(X, y)
    plt.title('Adaline - Versicolour classifier')
    ada2.plot_decision_regions(X, y)

    y = [-1]*100 + [1]*50
    ada3 = AdalineGD(eta=0.0001, n_iter=100, random_state=1)
    ada3.fit(X, y)
    plt.title('Adaline - Virginica classifier')
    ada3.plot_decision_regions(X, y)

    # Learning rate
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.show()

if __name__ == '__main__':
    main()
