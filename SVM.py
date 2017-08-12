import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def linear_kernel(X):
    return np.dot(X,X.T)
    
    
def gaussian_kernel(X):
    m = np.size(X, axis=0)
    kernel = np.zeros([m,m])
    linear = np.dot(X,X.T)
    for i in range(0,m):
        for j in range(0,m):
            kernel[i,j] = np.exp(-(1/(2*self.sigma*self.sigma))* \
                            (linear[i,i]+linear[j,j]-2*linear[i,j]))
    return kernel



class svm(object):
    """Support Vector Machine.
       
       Parameters:
       """
    
    def __init__(self, tol, C, kernel=linear_kernel):
        
        self.tol = tol
        self.C = C
        self.kernel = kernel
        
        
              
    def objective_function(self, alpha, X, y):
        """Defines objective function for optimal margin 
            classifier dual problem"""
        
        return np.sum(alpha) - 0.5*np.linalg.multi_dot([(alpha*y), \
                self.kernel(X), (alpha*y)])
    
    
    
    def clipping(self, lower_endpoint, upper_endpoint, x):
        """Don't think we need this, as scipy optimisation function can 
            incorporate bounds"""
        if x < lower_endpoint:
            return lower_endpoint
        elif x > upper_endpoint:
            return upper_endpoint
        else:
            return x
    
    
    
    def SMO_algorithm(self, X, y, max_passes):
        """Performs the SMO algorithm repeatedly until convergence"""
        
        # Initialise alpha and b to zero
        m = np.size(X, axis=0)
        alpha = np.zeros(m)
        b = 0
        
        # passes will count the number of passes without a change in alpha
        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0
            
            # For each index i, if alpha[i] violates KKT condition, then choose 
            # a random j!=i, and use them for the SMO step.
            for i in range(0,m):
            
                # Defines error term for i'th entry
                E_i = (np.dot(self.kernel(X),(alpha*y)))[i] + b - y[i]
                if (y[i]*E_i < -self.tol and alpha[i]<self.C) or \
                    (y[i]*E_i > self.tol and alpha[i]>0):
                    j = np.random.choice(np.delete(np.arange(0,m),i))
                    E_j = (np.dot(self.kernel(X),(alpha*y)))[j] + \
                        b - y[j]
                    old_alpha_i = alpha[i]
                    old_alpha_j = alpha[j]
                    
                    # Upper and lower clipping values
                    if abs(y[i]-y[j])<0.01:
                        lowerbound = np.amax([0, alpha[i] + alpha[j] - self.C])
                        upperbound = np.amin([self.C, alpha[i] + alpha[j]])
                    else:
                        lowerbound = np.amax([0, alpha[j] - alpha[i]])
                        upperbound = np.amin([self.C, self.C + \
                                        alpha[j] - alpha[i]])
                        
                    if abs(lowerbound - upperbound) < 0.000001:
                        continue
                    
                    eta = 2*self.kernel(X)[i,j] - self.kernel(X)[i,i] - \
                            self.kernel(X)[j,j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha[j]
                    alpha[j] = self.clipping(lowerbound, upperbound, alpha[j] -\
                                (1./eta)*y[j]*(E_i - E_j))
                    if abs(alpha[j] - old_alpha_j)<=0.000001:
                        continue
                    
                    # Update alpha[i]
                    alpha[i] = alpha[i] + y[i]*y[j]*(old_alpha_j - alpha[j])
                    #print "new alphai = ", alpha[i]
                    
                    # Choose b to satisfy the KKT conditions
                    if 0 < alpha[i] < self.C:
                        b += -E_i - y[i]*(alpha[i] - old_alpha_i)* \
                            self.kernel(X)[i,i] - y[j]*(alpha[j] - \
                            old_alpha_j)*self.kernel(X)[i,j]
                    elif 0 < alpha[j] < self.C:
                        b += -E_j - y[i]*(alpha[i] - old_alpha_i)* \
                            self.kernel(X)[i,j] - y[j]*(alpha[j] - \
                            old_alpha_j)*self.kernel(X)[j,j]
                    else: 
                        b += -0.5*(E_i + E_j + y[i]*(alpha[i] - old_alpha_i)* \
                            (self.kernel(X)[i,i] + self.kernel(X)[i,j]) \
                            + y[j]*(alpha[j] - old_alpha_j)* \
                            (self.kernel(X)[i,j] + self.kernel(X)[j,j]))
                            
                    num_changed_alphas += 1
            # If alpha wasn't changed, then pass over again, until max_passes is
            # reached
            if num_changed_alphas == 0:
                passes = passes + 1
            else: 
                passes = 0
        
        return alpha, b

