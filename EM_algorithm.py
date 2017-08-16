import numpy as np
import matplotlib.pyplot as plt
import k_means

class mix_of_gaussians(object):
    """ Mixture of Gaussians model. 
        The initial centres are found using k-means, and the initial variance 
        matrices are the covariance matrices of the points belonging to each 
        k-means cluster. After this, the EM algorithm is performed.
    
    Parameter:
    N       = number of Gaussians to fit data to
    tol     = numerical tolerance, determining when EM step has converged
    k_init  = number of initialisations of k-means to perform during
                initialisation
    n_init  = number of initialisations to perform"""
    
    
    def __init__(self, N, tol, k_init, n_init):
        self.N = N
        self.tol = tol
        self.k_init = k_init
        self.n_init = n_init
    
    
    def initialise_gaussians(self, X):
        """Initialises the N Gaussians, using k-means to find the initial means
            and setting the covariance matrices equal to the variance matrix of 
            the data points closest to each k-means cluster."""
        
        m = np.size(X, axis=0)
        n = np.size(X, axis=1)
        
        kmean = k_means.kmeans(X, self.N, self.k_init)
        initial = kmean.kmeans_alg()
        
        # Define covariance matrix for each cluster
        sigmas = np.zeros([self.N,n,n])
        
        # Set equal to covariance of data points allocated to each cluster
        for j in range(0, self.N):
            sigmas[j,:,:] = np.cov(X[(initial[1]==j).T[0],:].T)
        
        return initial[0], sigmas
        
        
    
    
    def gaussian_pdf(self, x, mu, Sigma, detSigma, invSigma):
        """PDF of multivariate Gaussian RV with mean mu and variance Sigma.
            Requires determinant and inverse of covariance matrix to be 
            calculated externally, so they are not caluculated repeatedly 
            during loops"""
        
        n = np.size(x)
        return (1./(detSigma*(2*np.pi)**(0.5*n)))* \
            np.exp(-0.5*np.dot((x-mu),np.dot(invSigma,(x-mu))))
            
            
        
    def cost_function(self, X, weights, phi, mus, Sigmas):
        """Cost function for EM algorithm""" 
        
        m = np.size(X, axis=0)
        cost = 0
        for j in range(0,self.N):
        
            # Define determinant and inverses here, to avoid recalculating for 
            # each loop
            det_Sigma_j = np.linalg.det(Sigmas[j,:,:])
            inv_Sigma_j = np.linalg.pinv(Sigmas[j,:,:])
            
            for i in range(0,m):
                cost += weights[i,j]*np.log(phi[j]* \
                    self.gaussian_pdf(X[i,:], mus[j,:], Sigmas[j,:,:], \
                    det_Sigma_j, inv_Sigma_j)/weights[i,j])
                        
        return cost
    
    
    
    def e_step(self, X, phi, mus, Sigmas):
        """E step - Updates the weights, indicating the probability of each 
            data point i belonging to cluster j"""
        
        m = np.size(X, axis=0)
        n = np.size(X, axis=1)
        weights = np.zeros([m, self.N])
        norm_factor = np.zeros([m,1])
        
        for j in range(0,self.N):
            # Define determinant and inverses here, to avoid recalculating for 
            # each loop
            det_Sigma_j = np.linalg.det(Sigmas[j,:,:])
            inv_Sigma_j = np.linalg.pinv(Sigmas[j,:,:])
            
            for i in range(0,m):
                term_j = phi[j]*self.gaussian_pdf(X[i,:], mus[j,:], \
                    Sigmas[j,:,:], det_Sigma_j, inv_Sigma_j)
                norm_factor[i] += term_j
                weights[i, j] += term_j
               
        for i in range(0,m):
                weights[i,:] = weights[i,:]/norm_factor[i]
                
        return weights
    
    
    
    def m_step(self, X, W, mus):
        """M step - Updates the parameters phi and (mu, Sigma) for each 
            Gaussian, based on the new weights"""
    
        m = np.size(X, axis=0)
        n = np.size(X, axis=1)
        new_phi = (1./m)*np.sum(W, axis=0)
        norm_factors = 1./(m*new_phi)
        #new_mus = np.dot((norm_factors*weight.T),X)
        #new_Sigmas = 
        
        new_mus = np.zeros([self.N, n])
        new_Sigmas = np.zeros([self.N, n, n])
        for j in range(0,self.N):
            #new_mus[j,:] = norm_factors[j]*np.dot(X.T, W[:,j])
            for i in range(0,m):
                new_mus[j,:] += W[i,j]*X[i,:]
                new_Sigmas[j,:,:] += W[i,j]* \
                    np.outer(X[i,:]-mus[j,:],X[i,:]-mus[j,:])
            new_mus[j,:] = norm_factors[j]*new_mus[j,:]
            new_Sigmas[j,:,:] = norm_factors[j]*new_Sigmas[j,:,:]
        
        return new_phi, new_mus, new_Sigmas
        
        
        
    def EM_main(self, X):
    
        m = np.size(X, axis=0)
        
        best_weights = (1./self.N)*np.ones([m, self.N])
        best_phi = (1./self.N)*np.ones([self.N,1])
        best_mus, best_Sigmas = self.initialise_gaussians(X)
        best_cost = self.cost_function(X, best_weights, best_phi, best_mus, \
                                               best_Sigmas)
        
        for i in range(0, self.n_init):
            
            old_weights = (1./self.N)*np.ones([m, self.N])
            old_phi = (1./self.N)*np.ones([self.N,1])
            old_mus, old_Sigmas = self.initialise_gaussians(X)
            old_cost = self.cost_function(X, old_weights, old_phi, \
                            old_mus, old_Sigmas)
        
            new_weights = self.e_step(X, old_phi, old_mus, old_Sigmas)
            new_phi, new_mus, new_Sigmas = self.m_step(X, new_weights, old_mus)
            new_cost = self.cost_function(X, new_weights, new_phi, new_mus, \
                                                new_Sigmas)
            
            while abs(new_cost - old_cost) > self.tol:
                old_weights = new_weights
                old_phi = new_phi
                old_mus = new_mus
                old_Sigmas = new_Sigmas
                old_cost = new_cost
                
                new_weights = self.e_step(X, old_phi, old_mus, old_Sigmas)
                new_phi, new_mus, new_Sigmas = self.m_step(X, new_weights, \
                                                                old_mus)
                new_cost = self.cost_function(X, new_weights, \
                                                new_phi, new_mus, new_Sigmas)
        
            if new_cost > best_cost:
                best_cost = new_cost
                best_weights = new_weights
                best_phi = new_phi
                best_mus = new_mus
                best_Sigmas = new_Sigmas
        
               
        return best_phi, best_mus, best_Sigmas
        
        
        
    def data_pdf(self, x, phi, mus, Sigmas):
    
        datapdf = 0
        for j in range(0, self.N):
            det_Sigma_j = np.linalg.det(Sigmas[j])
            inv_Sigma_j = np.linalg.pinv(Sigmas[j])
            datapdf += phi[j]*self.gaussian_pdf(x, mus[j,:], Sigmas[j,:,:], \
                        det_Sigma_j, inv_Sigma_j)
    
        return datapdf
            
        
    
