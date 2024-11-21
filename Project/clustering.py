from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def kmean(list, n_clusters = 4, random_state = 0):
    """
    Method to create K-means clustering

    Two variables:

    Number of clusters (n_clusters) - The number of clusters to create.
    Effect:
        A smaller n_clusters value results in fewer, larger clusters, potentially combining distinct groups.
        A larger n_clusters value allows finer granularity but may lead to over-segmentation.

    Random state (random_state) - Determines the random number generation for initializing cluster centroids.
    Effect:
        Setting a fixed random_state ensures reproducibility of clustering results.
        Using different random_state values may result in different clustering outputs due to randomness in centroid initialization.
    """

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit and transform the data with the scaler
    scaled_features = scaler.fit_transform(list)

    # Random nr of clusters
    kmeans = KMeans(n_clusters = n_clusters, random_state = random_state).fit(scaled_features)
    kmeans_labels = kmeans.fit_predict(scaled_features)

    return kmeans_labels

def dbScanning(list, eps = 0.15, min_samples = 10):
    """
    Method to create the DBscan clustering

    Two variables:

    Epsilon (eps) - The maximum distance between two points for them to be considered as part of the same "neighborhood."
    Effect:
        If eps is too small, many points might be classified as noise because they won't meet the neighborhood criteria.
        If eps is too large, clusters may merge, reducing the algorithm's ability to detect smaller distinct clusters.

    Mininum samples (min_samples) - The minimum number of points required to form a dense region (i.e., a cluster).
    Effect:
        A larger min_samples value requires denser clusters and is more conservative in forming clusters.
        A smaller min_samples value can lead to more clusters, potentially including outliers as valid clusters.
    """
    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit and transform the data with the scaler
    scaled_features = scaler.fit_transform(list)

    # Create a DBSCAN instance
    dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(scaled_features)

    # Fit the DBSCAN model
    dbscan_labels = dbscan.fit_predict(scaled_features)

    return dbscan_labels


def visualize_clusters(combined_features, labels, method='PCA'):
    """
    Visualize the clusters in 2D space.
    If features are >2 then using PCA to reduce dimensionality.
    """

    # Identify the clustering method
    if -1 in labels:
        clustering_method = 'DBSCAN'
    else:
        clustering_method = 'K-means'

    # Check if the combined features have 2 columns
    if combined_features.shape[1] != 2:
        print(
            f"Warning: The number of columns in 'combined_features' is {combined_features.shape[1]}. Performing dimensionality reduction.")

        if method == 'PCA':
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(combined_features)
            # Default labels for PCA components on plot
            xlabel, ylabel = 'Component 1', 'Component 2'
            print("x " + str(reduced_features[1].shape))
            print("y" + str(reduced_features[2].shape))
        else:
            raise ValueError("Method must be 'PCA'")
    else:
        # If there are already 2 columns, no dimensionality reduction is needed
        method = "None"
        reduced_features = combined_features.values
        # Labels used for the plot:
        ylabel, xlabel = combined_features.columns[0], combined_features.columns[1]

    # Plot the 2D scatter plot of the clusters with 'Year' on the x-axis and 'Rating_value' on the y-axis
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 1], reduced_features[:, 0], c=labels, cmap='viridis', alpha=0.6)
    plt.title(f'{clustering_method} Cluster Visualization with {ylabel} vs {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label='Cluster Label')
    plt.show()