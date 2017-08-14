import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def linear_kernel(x, y):
    return np.dot(x,y)
    
def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-(1/(2*sigma*sigma))*np.dot(x-y,x-y))
    
def polynomial_kernel(x, y, d=3.0):
    return (1. + np.dot(x,y)) ** d


class svm(object):
    """Support Vector Machine.
       
       Parameters:
       tol      = Tolerance for
       C        = Regularisation parameter
       kern     = Choice of kernel - linear_kernel or gaussian_kernel
                  (default = linear_kernel)
       """
    
    def __init__(self, tol, C, kern=linear_kernel):
        
        self.tol = tol
        self.C = C
        self.kern = kern
        
        
    def kernel(self,X):
        """Computes the Gram matrix for the data set"""
        m = np.size(X, axis=0)
        K = np.zeros([m,m])
        for i in range(0,m):
            for j in range(0,m):
                K[i,j] = self.kern(X[i,:], X[j,:])
        return K
    
              
    def objective_function(self, alpha, X, y):
        """Defines objective function for optimal margin 
            classifier dual problem.
            
            Parameters:
            alpha = Optimisation variable
            X = Gram matrix of data set
            y = Vector of classifications"""
        
        return np.sum(alpha) - 0.5*np.linalg.multi_dot([(alpha*y), self.kernel(X), (alpha*y)])
    
    
    
    def clipping(self, lower_endpoint, upper_endpoint, x):
        """Clips a value to lie within an interval.
        
        Parameters:
        lower_endpoint = Lower endpoint of interval
        upper_endpoint = Upper endpoint of interval
        x              = Value to be clipped"""
        
        if x < lower_endpoint:
            return lower_endpoint
        elif x > upper_endpoint:
            return upper_endpoint
        else:
            return x
    
    
    
    def SMO_algorithm(self, X, y, max_passes):
        """Performs the SMO algorithm repeatedly until convergence.
        
        Parameters: 
        X           = Gram matrix for data set
        y           = Vector of classifications for data points
        max_passes  = Maximum number of times to repeat SMO step, 
                      without changes to the optimisation variable."""
        
        # Initialise alpha and b to zero
        m = np.size(X, axis=0)
        alpha = np.zeros(m)
        b = 0
        
        # Find kernel matrix
        K = self.kernel(X)
        
        # passes will count the number of passes without a change in alpha
        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0
            
            # For each index i, if alpha[i] violates KKT condition, then choose 
            # a random j!=i, and use them for the SMO step.
            for i in range(0,m):
            
                # Defines error term for i'th entry
                E_i = (np.dot(K,(alpha*y)))[i] + b - y[i]
                if (y[i]*E_i < -self.tol and alpha[i]<self.C) or \
                    (y[i]*E_i > self.tol and alpha[i]>0):
                    j = np.random.choice(np.delete(np.arange(0,m),i))
                    E_j = (np.dot(K,(alpha*y)))[j] + \
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
                    
                    eta = 2*K[i,j] - K[i,i] - \
                            K[j,j]
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
                            K[i,i] - y[j]*(alpha[j] - \
                            old_alpha_j)*K[i,j]
                    elif 0 < alpha[j] < self.C:
                        b += -E_j - y[i]*(alpha[i] - old_alpha_i)* \
                            K[i,j] - y[j]*(alpha[j] - \
                            old_alpha_j)*K[j,j]
                    else: 
                        b += -0.5*(E_i + E_j + y[i]*(alpha[i] - old_alpha_i)* \
                            (K[i,i] + K[i,j]) \
                            + y[j]*(alpha[j] - old_alpha_j)* \
                            (K[i,j] + K[j,j]))
                            
                    num_changed_alphas += 1
                    
            # If alpha wasn't changed, then pass over again, until max_passes is
            # reached
            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0      
        
        return alpha, b



def classifier_function(X, y, v, optimal_values, kernel=linear_kernel):
    """Given a data point, calculate the classifier function.
    
    Parameters: 
    X               = Gram matrix of data set
    y               = Vector of classifications for data points
    v               = Point to evaluate classifer at
    optimal_values  = Optimal parameters found during SMO
    kernel          = Choice of kernel function (default = linear_kernel)"""
    
    # Split optimal_values into vector and intercept terms
    alpha = optimal_values[0]
    b = optimal_values[1]
    
    # Evaluate classifier function at v
    m = np.size(X, axis=0)
    f = b
    for k in range(0,m):
        f += alpha[k]*y[k]*kernel(X[k,:],v)
    return f
    
    

def plot_decision_boundary(optimal_values, X, y, kernel=linear_kernel):
    """Once the dual problem has been solved, plot the decision boundary
        (only for 2-D data)
    
    Parameters:
    alpha   = Optimal value for dual problem
    b       = Intercept term
    X       = Gram matrix for data set 
    y       = Vector of classifications for data points
    x_range = Interval describing range of first feature
    y_range = Interval describing range of second variable
    kernel  = Choice of kernel when calculating alpha"""
    
    # Find ranges of both features, to find required size of mesh
    x1_lower, x2_lower = np.amin(X, axis=0)
    x1_higher, x2_higher = np.amax(X, axis=0)
    
    # Create mesh over data range
    meshx, meshy = np.meshgrid(np.linspace(x1_lower, x1_higher,50), \
                                np.linspace(x2_lower, x2_higher,50))
    
    # Calculate classifier over mesh
    m = np.size(X, axis=0)
    z = np.zeros(np.shape(meshx))
    for i in range(0,np.size(meshx, axis=0)):
        for j in range(0,np.size(meshx, axis=1)):
            v = np.array([meshx[i,j],meshy[i,j]])
            z[i,j] = classifier_function(X, y, v, optimal_values, kernel)
    
    # Plot data on new plot
    plt.figure()
    for i in range(0,np.size(X,axis=0)):
        if y[i]<0:
            plt.scatter(X[i,0],X[i,1],c='b')
        else:
            plt.scatter(X[i,0],X[i,1],c='r')
    
    # Plot contour for zero set of classifier function
    plt.contour(meshx, meshy, z, levels=[0.0])
    plt.show()



