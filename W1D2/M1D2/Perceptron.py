#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Assignment submisstion
# Albus & Han
"""
Implement the Preceptron Algorithm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """Perceptron classifier.

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
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.errors_ = []

    def fit(self, X, y):
        """Fit training data.

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
            prediction = self.predict(X)
            results = y - prediction
            self.errors_.append(np.sum(np.abs(results))/2)

            delta = np.asmatrix(results)*with_bias
            delta *= self.eta
            self.w_ += np.squeeze(np.asarray(delta))

    def net_input(self, X):
        """Calculate net input"""
        with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(with_bias, self.w_)
        

    def predict(self, X):
        """Return class label after unit step"""
        return np.sign(self.net_input(X))


def main():

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

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')

    plt.show()

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

if __name__ == '__main__':
    main()
