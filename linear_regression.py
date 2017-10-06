import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class linear_regressor(object):
    """Linear regressor.
       
       Parameters:
       alpha = learning rate
       iterations = number of iterations of gradient descent"""
    
    def __init__(self, X, y, alpha, iterations):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.iterations = iterations
    
    def cost_function(self, Theta):
        m = np.size(self.X, axis=0)
        X2 = np.column_stack([np.ones(self.y.size), self.X])
        return (1./m)*np.inner((np.dot(X2,Theta)-self.y),(np.dot(X2,Theta)-self.y))
    
    def gradient_descent(self, Theta, plot = None):
        m = self.y.size
        X2 = np.column_stack([np.ones(m), self.X])
        for i in np.arange(1,self.iterations):
            Theta += self.alpha*np.dot((self.y - np.dot(X2, Theta)), X2)    
        if plot == 'on':
            plt.scatter(self.X, self.y)
            plt.plot(np.arange(np.amin(self.X), np.amax(self.X), 0.1), \
                     (Theta[0] + Theta[1]*np.arange(np.amin(self.X), np.amax(self.X), 0.1)),color='r')
            plt.show()
        return Theta
