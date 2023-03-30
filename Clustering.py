import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering

def k_means_clustering(file, start):
    dataset = pd.read_csv('Data Directory/clustering.csv')
    X = dataset.iloc[:, [start, start+1]].values

    # Using Elbow method to find optimal number of clusters
    wcss = []
    for x in range(1, 11):
        k_means = KMeans(n_clusters=x, init='k-means++', random_state=42, n_init=10)
        k_means.fit(X)
        wcss.append(k_means.inertia_)

    # Visualization of graph
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('Images/K-Means Clustering')
    plt.show()

    # NOTE: This plot on this data shows the elbow at 5 clusters

    # Training the K-Means model on the data set
    print(k_means)
    print()
    k_means = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    y_kmeans = k_means.fit_predict(X)

    print(k_means)

    # Visualizing these clusters
    print(X[y_kmeans == 0, 0])
    print(X[y_kmeans == 0, 1])
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters of customers (K-Means)')
    plt.xlabel('Annual income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.savefig('Images/Clusters_K-Means.png')
    plt.show()

def hierarchical_clustering(file, start):
    dataset = pd.read_csv(file)
    X = dataset.iloc[:, [start, start+1]].values

    # Using dendogram to find optimal clusters
    dendogram = sch.dendrogram(sch.linkage(X, method='ward'))

    # Visualize dendogram
    plt.title('Dendogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean Distances')
    plt.show()

    # Train hierarchical clustering model on dataset
    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)

    # Visualize the clusters
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    for x in range(5):
        temp = 'Cluster ' + str((x+1))
        plt.scatter(X[y_hc == x, 0], X[y_hc == x, 1], s = 100, c = colors[x], label = temp)
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending score (1-100)')
    plt.legend()
    plt.show()