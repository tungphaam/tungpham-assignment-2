import numpy as np

class KMeans:
    def __init__(self, k, max_iters=100, tolerance=1e-4, init_method='random'):
        """
        Initializes the KMeans object with the number of clusters, max iterations, tolerance, and initialization method.
        """
        self.k = k  
        self.max_iters = max_iters  
        self.tolerance = tolerance  
        self.centroids = None  
        self.init_method = init_method 
        self.steps = [] 

    def fit(self, data):
        self._validate_k(data)
        self._initialize_centroids(data)

        for _ in range(self.max_iters):
            clusters = self._assign_clusters(data)
            new_centroids = self._calculate_new_centroids(data, clusters)
            
            # Store the state of the centroids and clusters at each step
            self.steps.append({
                'centroids': new_centroids.copy(),
                'clusters': clusters.copy()
            }) 

            if np.all(np.abs(new_centroids - self.centroids) <= self.tolerance):
                break

            self.centroids = new_centroids

    def predict(self, data):
        # Predict which cluster each data point belongs to
        clusters = self._assign_clusters(data)
        return clusters
    
    def get_steps(self):
        return self.steps
    
    def _validate_k(self, data):
        # Validates the number of clusters (k)
        if self.k < 2 or self.k > len(data):
            raise ValueError(f"Invalid number of clusters: 'k' must be between 2 and {len(data)}.")
        
    def _initialize_centroids(self, data):
        # Initializes the centroids based on the selected intialization method
        if self.init_method == 'random':
            self.centroids = data[np.random.choice(len(data), self.k, replace=False)]
            print(f"Random centroids: {self.centroids}")
        elif self.init_method == 'farthest_first':
            self.centroids = self._initialize_farthest_first(data)
            print(f"Farthest first centroids: {self.centroids}")
        elif self.init_method == 'kmeans++':
            self.centroids = self._initialize_kmeans_pp(data)
            print(f"KMeans++ centroids: {self.centroids}")

        if self.centroids is None or len(self.centroids) != self.k:
            raise ValueError("Centroids were not initialized correctly.")

    def _assign_clusters(self, data):
        # Assign each data point to the closest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_new_centroids(self, data, clusters):
        # Calculate new centroids based on current cluster assignments
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(self.k)])
        return new_centroids  
    
    def _is_converged(self, old_centroids, new_centroids):
        # Checks if the centroids have converged
        return np.all(np.abs(new_centroids - old_centroids) <= self.tolerance)

    def _initialize_random(self, data):
        # Random initialization by choosing k random points
        random_indices = np.random.choice(len(data), self.k, replace=False)
        return data[random_indices]

    def _initialize_farthest_first(self, data):
        # Farthest First initialization
        centroids = [data[np.random.choice(len(data))]]  # Start with a random point
        for _ in range(1, self.k):
            # Compute distances from current centroids to all data points
            distances = np.min([np.linalg.norm(data - c, axis=1) for c in centroids], axis=0)
            # Choose the farthest point from the current centroids
            next_centroid = data[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _initialize_kmeans_pp(self, data):
        # KMeans++ initialization
        centroids = [data[np.random.choice(len(data))]]  # Start with a random point
        for _ in range(1, self.k):
            # Compute squared distances from the closest centroid
            distances = np.min([np.linalg.norm(data - c, axis=1) ** 2 for c in centroids], axis=0)
            # Probabilistically choose the next centroid based on squared distances
            probabilities = distances / np.sum(distances)
            next_centroid = data[np.random.choice(len(data), p=probabilities)]
            centroids.append(next_centroid)
        return np.array(centroids)


if __name__ == "__main__":
    # Generate random data points
    data = np.random.uniform(-10, 10, (300, 2))

    kmeans = KMeans(k=3, init_method='kmeans++')
    kmeans.fit(data)
    clusters = kmeans.predict(data)

    print("Centroids: ", kmeans.centroids)
    print("Cluster Assignments: ", clusters)
