import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score
from scipy.cluster.vq import kmeans,vq

'''
@caelana
script for playing around with clustering and scikit
'''

x, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

def getNumberOfClusters(x, max_clusters):
    '''
    Get number of clusters using sillhouette method
    Suitable only for datasets with 2+ clusters
    '''
    sil = []
    
    try:
        for k in range(2, max_clusters):
            kmeans = KMeans(n_clusters = k).fit(x)
            labels = kmeans.labels_
            sil.append(silhouette_score(x, labels, metric = 'euclidean'))
            
        return np.argmax(sil) + 2
    except:
        return 0

kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

for i in range(len(centers)):
  print("Cluster", i)
  print("Center:", centers[i])
  print("Size:", sum(kmeans.labels_ == i))
  
plt.show()


