import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class logistic_regressor(object):
    """Logistic regressor.
       
       Parameters:
       alpha = learning rate
       iterations = number of iterations of gradient descent"""
    
    def __init__(self, X, y, alpha, iterations):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.iterations = iterations
        
    def logistic(self,t):
        return 1.0/(1+np.exp(-t))
    
    def kernel(self,u,v,sigma):
        return np.exp((1./(2*sigma*sigma))*np.diagonal(np.dot((u-v),np.transpose(u-v))))
    
    def cost_function(self, Theta, sigma):
        X2 = np.column_stack([np.ones(self.y.size), self.X])
        return sum(self.y*np.log(self.logistic(self.kernel(X2,Theta,sigma))) + \
                   (1-self.y)*np.log(1-self.logistic(self.kernel(X2,Theta,sigma))))
    
    def gradient_descent(self, Theta, sigma, plot = None):
        m = self.y.size
        X2 = np.column_stack([np.ones(m), self.X])
        for i in np.arange(1,self.iterations):
            Theta += self.alpha*np.dot((self.y - self.logistic(self.kernel(X2, Theta, sigma))), X2)
        if plot == 'on':
            plt.scatter(self.X, self.y)
            plt.plot(np.arange(self.X[0], self.X[-1], 0.1), \
                     self.logistic(Theta[0] + Theta[1]*np.arange(self.X[0], self.X[-1], 0.1)))
            plt.show()
        return Theta


#m = y.size
#X2 = np.column_stack([np.ones(m),X])
#X2 = X2.astype(np.float64)

#def logistic(t):
#    return 1.0/(1+np.exp(-t))
#
#def cost_function(X,y,Theta):
#    return sum(y*np.log(logistic(np.dot(X,Theta))) + (1-y)*np.log(1-logistic(np.dot(X,Theta))))
#    
#def gradient_descent(X,y,Theta,alpha,iterations):
#    u = [0]
#    v = []
#    for i in np.arange(1,iterations):
#        Theta = Theta + alpha*np.dot((y-logistic(np.dot(X,Theta))),X2)
#        u.append(u[-1]+1)
#        v.append(cost_function(X,y,Theta))
#    return Theta, u[1:], v

#Theta = np.zeros(2)
#Theta, u, v = gradient_descent(X2,y,Theta,alpha,iterations)

#plt.plot(u,v)
#plt.show()

#plt.scatter(X,y)
#plt.plot(np.arange(X[0],X[-1],0.1), logistic(Theta[0]+Theta[1]*(np.arange(X[0],X[-1],0.1))),'r') 
#plt.show()
