import time
import pandas as pd
import numpy as np
import preprocess as pp
import TF_IDF as freq
import clustering as cl

def for_demonstration():
    start_time = time.time()

    # Webscraping part
    # fetch_titles()
    # movies = pd.read_csv('movie_titles_and_ids.csv')
    # get_genre_n_production(movies)

    # Load the file in
    movies = pd.read_csv('movies_with_genres.csv').copy()

    # Clean the file (remove series, elements with N/A and split rating into:
    # Rating value
    # Votes
    print(len(movies))
    movies = pp.clean_tits(movies)
    print(len(movies))

    # Convert the 'Year' column to numeric
    movies['Year'] = pd.to_numeric(movies['Year'], errors='coerce')

    # Creating clusers based on rating_value and year:
    numerical_features = movies[['Year', 'Rating_value', 'votes']]

    # TF-IDF
    tfidf_scores = freq.tfidf_values(movies)

    # Creating a new frame:
    combined_features = np.hstack((tfidf_scores, numerical_features.values))

    # k-means
    kmeans_labels = cl.kmean(combined_features)

    # DBscan
    # dbscan_lambels = cl.dbScanning(numerical_features)

    # Visualize the clusters using PCA (or t-SNE)
    cl.visualize_clusters(numerical_features, kmeans_labels, method='PCA')
    # cl.visualize_clusters(numerical_features, dbscan_lambels, method='PCA')

    print("--- %s seconds ---" % (time.time() - start_time))
