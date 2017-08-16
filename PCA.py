import numpy as np
import matplotlib.pyplot as plt

def data_normalisation(X):
    """ Normalises a data set, so that each feature has zero mean and
        unit variance."""
        
    m = np.size(X, axis=0)
    
    # Centre mean
    mu = (1./m)*np.sum(X, axis=0)
    normalised_X = X-mu
    
    # Scale variances
    sigma2 = (1./m)*np.diag(np.dot(normalised_X.T,normalised_X))
    normalised_X = normalised_X/np.sqrt(sigma2)
    
    return normalised_X
    
 
 
def pca(X, N):
    """ Performs PCA on a data set, projecting points onto the N-dimension 
        subspace with largest variance. Note: assumes data set has been 
        normalised, so each feature has zero mean and unit variance.
    
        Parameters:
        X = data set
        N = dimension of suspace to project data onto"""
        
    m = np.sum(X, axis=0)

    # Construct covariance matrix
    Sigma = (1./m)*np.dot(X.T,X)
    
    # Find N principal eigenvectors of Sigma
    basis_vectors = np.linalg.eigh(Sigma)[1][-N:]
    
    # Project data points onto subspace
    return np.dot(X, basis_vectors.T)
    
