import numpy as np
import matplotlib.pyplot as plt


class k_means_clustering:
  def __init__(self, data, search_number_of_clusters=True, clusters=10, iterations=20):
    self.search_number_of_clusters = search_number_of_clusters
    if self.search_number_of_clusters:
      self.cost = []
    else:
      self.cost = 0

    self.clusters = clusters
    
    self.iterations = iterations
    self.data = data
  
  def init_centroids(self, clusters):
    centroids = np.zeros((clusters, len(self.data[0,:])))
    centroids_idx = np.random.choice(self.data.shape[0], clusters, replace=False)
    for i, idx in enumerate(centroids_idx):
      centroids[i, :] = self.data[idx]
    
    return centroids
  
  def find_closest_centroid(self, clusters, centroids):
    m = self.data.shape[0]
    idx = np.zeros((m,1),dtype=int)

    for i in range(0, m):
      idx[i] = 0
      for d in range(0, clusters):
        distance = np.sum(np.square(self.data[i,:] - centroids[d,:]))
        if distance <= np.sum(np.square(self.data[i,:] - centroids[idx[i],:])):
          idx[i] = d
    
    return idx
  
  def compute_centroids(self, idx, clusters, centroids):
    new_centroids = np.zeros((clusters, self.data.shape[1]))

    for i in range(0, clusters):
      cluster_idx = (idx == i) * 1
      cluster_points = np.multiply(self.data, cluster_idx)
      if np.sum(cluster_idx) > 0: 
        tot = np.sum(cluster_idx)
      else:
        tot = 1
      new_centroids[i, :] = (1/tot) * np.sum(cluster_points, 0)

    return new_centroids

  def fit(self):
    if self.search_number_of_clusters:
      for i in range(1, self.clusters+1):
        print("fitting for clusters: ", i)
        centroids = self.runKMeans(i)
        self.cost.append(self.compute_distortion(i, centroids))
      self.plot_costs()
    else:
      return self.runKMeans(self.clusters)
  
  def runKMeans(self, clusters):
     final_centroids =  np.zeros((clusters, self.data.shape[1]))
     for i in range(0, self.iterations):
       centroids = self.init_centroids(clusters)
       while True:
         idx = self.find_closest_centroid(clusters, centroids)
         new_centroids = self.compute_centroids(idx, clusters, centroids)
         if np.array_equal(centroids, new_centroids):
           centroids = new_centroids
           break
         else:
           centroids = new_centroids

       if self.compute_distortion(clusters, centroids) < self.compute_distortion(clusters, final_centroids):
        final_centroids = centroids
     
     return final_centroids 

  def compute_distortion(self, clusters, centroids):
    m = self.data.shape[0]
    idx = self.find_closest_centroid(clusters, centroids)

    distortion = 0
    for i in range(0, centroids.shape[0]):
      cluster_idx = (idx == i) * 1
      distortion += np.sum(np.multiply(np.square(self.data - centroids[i,:]), cluster_idx))

    return (1/m) * distortion

  def plot_costs(self):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(np.arange(self.clusters)+1, self.cost, figure=fig)
    plt.title('Algoritmo de K-Medias', fontsize=30)
    plt.xlabel('Número de Clusters', fontsize=20)
    plt.ylabel('Distorsión', fontsize=20)
    plt.savefig('clusters-dist.png')