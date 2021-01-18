import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# Compute the observation matrix
# from a matrix of correlations using method (c)
# where Xij = (1/2 (1- pij))^1/2
# and pij is the correlation between observation i and j
def observationMatrix(X):
  X = 1 - X
  X = X * 0.5
  return np.sqrt(X)

# For large matrices X, generally it is good practice to reduce its dimension via
# PCA. The idea is to replace X with its standardized orthogonal projection onto a
# lower-dimensional space, where the number of dimensions is given by the number
# of eigenvalues in X’s correlation matrix that exceed λ þ (see Section 2). The
# resulting observations matrix, X e , of size NxF, has a higher signal-to-noise ratio.
#
# s represents the number of values to be replaced
def replaceStandarized(X, s):
  return X[:, s:]


def clusterKMeansBase(corr0, maxNumClusters=10,n_init=10):
  x,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series()# observations matrix
  for init in range(n_init):
    for i in range(2,maxNumClusters+1):
      kmeans_=KMeans(n_clusters=i,n_jobs=1,n_init=1)
      kmeans_=kmeans_.fit(x)
      silh_=silhouette_samples(x,kmeans_.labels_)
      stat=(silh_.mean()/silh_.std(),silh.mean()/silh.std())
      if np.isnan(stat[1]) or stat[0]>stat[1]:
        silh,kmeans=silh_,kmeans_
  
  newIdx=np.argsort(kmeans.labels_)
  corr1=corr0.iloc[newIdx] # reorder rows
  
  corr1=corr1.iloc[:,newIdx] # reorder columns
  clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() \
    for i in np.unique(kmeans.labels_) } # cluster members
  silh=pd.Series(silh,index=x.index)
  
  return corr1,clstrs,silh
