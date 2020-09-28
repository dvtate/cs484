

import pandas
import sklearn.cluster as cluster

X = pandas.DataFrame({‘x’: [0.8, 0.3, 0.1, 0.4, 0.9]})
myCluster = cluster.KMeans(n_clusters = 2, random_state = 0).fit(X)
print(“Cluster Assignment:”, myCluster.labels_)
print(“Cluster Centroid 0:”, myCluster.cluster_centers_[0])
print(“Cluster Centroid 1:”, myCluster.cluster_centers_[1])