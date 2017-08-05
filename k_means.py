import numpy as np

class kmeans():
    """K-means clustering algorithm.
    
       parameter: 
       X = data points
       N = number of clusters
       n_init = number of initialisations"""
       
       
    def __init__(self, X, N, n_init):
        self.X = X
        self.N = N
        self.n_init = n_init
        
        
        
    def initialise(self):
        """ Initialise the centroids, as N random points from data set"""
        
        m = np.size(self.X, axis=0)
        num_features = np.size(self.X, axis=1)
        
        # Set each centroid to random data point
        random_datapoint_indices = np.random.choice(m, self.N, replace=False)
        return self.X[random_datapoint_indices,:]
           
            
          
    def distortion_function(self, centroids, centroid_labels):
        """Distortion function for k-means, used to determine local minima"""
        
        return np.linalg.norm(np.ndarray.flatten(self.X - \
                centroids[np.transpose(centroid_labels).astype(int),:]))
                  
            
            
    def assignment_step(self, centroids):
        """ Assigment step: assign each data point to nearest centroid"""
        
        # Define number of data points and centroif label matrix
        m = np.size(self.X, axis = 0)
        centroid_labels = np.zeros([m,1])
        
        # For each data point, find index of centroid from 
        # centroids closest to data point
        for i in range(0, m):
            data_point = self.X[i,:]
            min_centroid_index = np.argmin(np.linalg.norm(centroids - \
                data_point, axis=1))
            centroid_labels[i] = min_centroid_index 
        # Return list of centroid labels closest to each data point
        return centroid_labels
       
      
      
    def update_step(self, centroid_labels):
        """ Update step: move each centroid to mean of data points 
            assigned to it"""
        
        # Define new centroids matrix                
        m = np.size(self.X, axis=0)
        num_features = np.size(self.X, axis=1)    
        centroids = np.zeros([self.N,num_features])
        
        # For each centroid, move to mean of data points assigned to it
        for j in range(0,self.N):
            centroids[j,:] = (1./np.sum(centroid_labels==j))* \
                np.sum((centroid_labels==j)*self.X, axis=0)
        
        # Return new centroids
        return centroids
        
        
        
    def kmeans_alg(self):
        """ Performs k-means algorithm on the data set, performing n_iter 
            iterations to find the centroids with lowest distortion"""
        
        # Initialise to find base distortion level
        best_centroids = self.initialise()
        centroid_labels = self.assignment_step(best_centroids)
        best_distortion = self.distortion_function(best_centroids, \
            centroid_labels)
        
        
        for i in range(self.n_init):
            # Initialise centroids
            centroids = self.initialise()
            
            # Perform assignment and update steps. Repeat until centroids
            # remain the same
            old_centroids = 0.
            while np.linalg.norm(centroids - old_centroids)>0:
                old_centroids = centroids
                centroid_labels = self.assignment_step(centroids)
                centroids = self.update_step(centroid_labels)
                
            # Compare distortion of new centroids to previous iterations
            distortion = self.distortion_function(centroids,centroid_labels)
            if distortion < best_distortion:
                best_centroids = centroids
                best_distortion = distortion
        
        return best_centroids    
