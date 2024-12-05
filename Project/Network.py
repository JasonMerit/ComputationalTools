from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as net
import math
import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Compute Weighted Rating (WR)
def Bayesian_weighted_average(movies, n):
    """
        get n most popular movies using Bayesian weighted average based on rating and votes:
        WR = v/(v + m) * R + m/(v + m) * C

        where:
        R: Average rating of the movie (Rating_value).
        v: Number of votes for the movie (votes).
        m: Minimum votes required to be considered (e.g., 80th percentile of votes).
        C: Mean rating across all movies.
    """

    C = movies['Rating_value'].mean()

    if n < len(movies):
        q = 1 - (n / len(movies))
        m = movies['votes'].quantile(q)  # 80th percentile as threshold
    else:
        m = 1

    # Filter movies with at least `m` votes
    qualified_movies = movies[movies['votes'] >= m]

    # Compute Weighted Rating (WR)
    def weighted_rating(x, m=m, C=C):
        v = x['votes']
        R = x['Rating_value']
        return (v / (v + m) * R) + (m / (v + m) * C)

    qualified_movies['score'] = qualified_movies.apply(weighted_rating, axis=1)

    # Sort by Weighted Rating and select Top n
    top_movies = qualified_movies.sort_values('score', ascending=False).head(n)

    # reset the index
    top_movies.reset_index(drop=True, inplace=True)

    return top_movies

def show_distrubtion(top_movies):
    """
    Plots the distribution of how the movies are assigned to the different clusters in:
    summary plot, Genre and Directors/stars
    """
    # Set the style for the plots
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 20))

    # Plot for clusters
    plt.subplot(3, 1, 1)
    sns.countplot(x='Description', data=top_movies)
    plt.xlabel('Plot summary clusters')
    plt.ylabel('Number of movies')
    plt.title("Distribution of movie clusters by plot summaries")
    plt.xticks(rotation=0)

    # Plot for categories
    plt.subplot(3, 1, 2)
    sns.countplot(x='Genre', data=top_movies)
    plt.xlabel('Genre cluster')
    plt.ylabel('Number of movies')
    plt.title("Distribution of movie clusters by Genres")
    plt.xticks(rotation=90)

    # Plot for categories
    plt.subplot(3, 1, 3)
    sns.countplot(x='Directors', data=top_movies)
    plt.xlabel('Director and stars combined cluster')
    plt.ylabel('Number of movies')
    plt.title("Distribution of movie clusters by Directors and Stars combined")
    plt.xticks(rotation=90)

    plt.savefig("distribution_of_clusters.png", format="PNG", bbox_inches="tight")  # Save the plot to a file

    plt.show()

def plot_graph(graph):
    """
    Plots the graph using NetworkX and Matplotlib.
    """
    # Set up the figure size
    plt.figure(figsize=(15, 15))

    # Prepare node sizes, colors, and labels
    sizes = []
    colors = []
    label_vec = {}

    # Iterate through the nodes to apply sizes, colors, and labels
    for node, attributes in graph.nodes(data=True):
        if 'label' in attributes and attributes['label'] == 'Movie':
            # Use a Bayesian weighted average for node size
            sizes.append(100 * attributes.get('score', 1))
            label_vec[node] = node  # Add the movie name to the labels
            colors.append('red')  # Color for movie nodes
        else:
            sizes.append(10)  # Default size for non-movie nodes (e.g., genres, directors)
            colors.append('green')  # Color for non-movie nodes

    # Layout of the graph (using Kamada-Kawai layout)
    pos = net.kamada_kawai_layout(graph)

    # Draw the graph
    net.draw(graph, pos=pos, node_size=sizes, node_color=colors, with_labels=False, alpha=0.6, width=0.5,
            edge_color='grey')

    # Draw the labels for all nodes (movies, directors, genres, etc.)
    net.draw_networkx_labels(graph, pos, labels=label_vec, font_size=12, font_color='black')

    # Highlight the movie nodes with a distinct color and size
    net.draw_networkx_nodes(graph, pos, nodelist=label_vec.keys(), node_size=200, node_color='red', edgecolors='black')

    # Show the plot
    plt.title("Movie Network with All Movie Labels")
    plt.axis('off')  # Hide axis
    plt.savefig("network_of_movies.png", format="PNG", bbox_inches="tight")  # Save the plot to a file
    plt.show()

def plot_graph_n_recommended(graph, recommended_list, seen_movie):
    """
    Highlight the recommended way on the graph from the seen movie
    """
    # Set up the figure size
    plt.figure(figsize=(15, 15))

    # Prepare node sizes, colors, and labels
    sizes = []
    colors = []
    label_vec = {}
    seen_movie_nodes = set([seen_movie])  # Set to track seen movie node
    recommended_movie_nodes = set([movie for movie, _ in recommended_list])  # Set for recommended movie nodes
    highlighted_edges = []  # List to track edges between the seen movie and recommended movies

    # Iterate through the nodes to apply sizes, colors, and labels
    for node, attributes in graph.nodes(data=True):
        if 'label' in attributes and attributes['label'] == 'Movie':
            label_vec[node] = node  # Add the movie name to the labels

            # Check if it's the seen movie or a recommended movie
            if node == seen_movie:
                sizes.append(500)  # Larger size for the seen movie
                colors.append('green')  # Color for the seen movie
            elif node in recommended_movie_nodes:
                sizes.append(400)  # Medium size for the recommended movies
                colors.append('yellow')  # Color for the recommended movies
            else:
                sizes.append(100)  # Default size for other movie nodes
                colors.append('red')  # Default color for non-highlighted movie nodes
        else:
            sizes.append(10)  # Default size for non-movie nodes (e.g., genres, directors)
            colors.append('grey')  # Color for non-movie nodes

    # Identify edges between the seen movie and the recommended movies
    for recommended_movie, _ in recommended_list:
        if graph.has_edge(seen_movie, recommended_movie):
            highlighted_edges.append((seen_movie, recommended_movie))

    # Layout of the graph (using Kamada-Kawai layout)
    pos = net.kamada_kawai_layout(graph)

    # Draw the graph
    net.draw(graph, pos=pos, node_size=sizes, node_color=colors, with_labels=False, alpha=0.6, width=0.5,
            edge_color='grey')

    # Draw the labels for all nodes (movies, directors, genres, etc.)
    net.draw_networkx_labels(graph, pos, labels=label_vec, font_size=12, font_color='black')

    # Highlight the seen movie in green and recommended movies in yellow
    net.draw_networkx_nodes(graph, pos, nodelist=seen_movie_nodes, node_size=600, node_color='green', edgecolors='black')
    net.draw_networkx_nodes(graph, pos, nodelist=recommended_movie_nodes, node_size=400, node_color='yellow',
                           edgecolors='black')

    # Highlight the edges between the seen movie and recommended movies in blue
    net.draw_networkx_edges(graph, pos, edgelist=highlighted_edges, edge_color='blue', width=2)

    # Show the plot
    plt.title("Movie Network with Seen and Recommended Movies Highlighted")
    plt.axis('off')
    plt.savefig("highlighted_network_of_movies.png", format="PNG", bbox_inches="tight")  # Save the plot to a file
    plt.show()

def plot_simpel_recommended_movies(recommended_movies, seen_movie):
    """
    Simple plot to showcase the recommended movies based and the adamic adar index based on the "seen_movie"
    """
    # Simpel plot of the recommended movies
    movie_names = [movie[0] for movie in recommended_movies]
    scores = [movie[1] for movie in recommended_movies]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(movie_names, scores, color='skyblue')

    # Set the labels and title
    plt.ylabel('Adamic Adar Score')
    plt.title(f'Recommended Movies Based on: {seen_movie}')

    # Rotate x-axis labels to make them readable
    plt.xticks(rotation=10, ha='right')

    plt.savefig("Adamic_adar_plot.png", format="PNG", bbox_inches="tight")  # Save the plot to a file

    # Show the plot
    plt.show()

def create_graph(movies, tf, top_n = 5):
    """
    Creates a network graph representing relationships between movies, directors/stars, genres,
    summary plot, and similar movies based on cosine similarity.

    The function uses the NetworkX library to build a graph where:
        - Nodes represent movies, directors/stars, genres, and summary plot.
        - Edges represent relationships, such as "directed by," "belongs to genre,"
          "has description," and "is similar to."

    Method:
        1. Adds nodes for movies with attributes such as name and Bayesian weighted score.
        2. Adds nodes for directors/stars, genres, and summary plot if they do not already exist.
        3. Connects movies to their directors/stars, genres, and descriptions via edges.
        4. Uses cosine similarity to find the `top_n` most similar movies for each movie,
           connecting them with labeled edges.
    """
    # Use network Analysis library NetworkX
    graph = net.Graph()

    # Create the graph
    for i, movie in movies.iterrows():
        # add the nodes (score is the calculated Bayesian weighted average based on rating_value and votes)
        graph.add_node(movie['Name'], key=movie['Name'], label = 'Movie', rating = movie['score'])

        # add nodes for the directors and movies
        if not graph.has_node(movie['Directors']):
            graph.add_node(movie['Directors'], label = 'Directors')

        # add genre and connect them
        if not graph.has_node(movie['Genre']):
            graph.add_node(movie['Genre'], label = 'Genre')

        if not graph.has_node(movie['Description']):
            graph.add_node(movie['Description'], label = 'Description')

        # add edges between movie names and summary plot
        # Add edges between Genre and Name
        # Add edges betwen Genre and Directors/stars
        graph.add_edge(movie['Name'], movie['Directors'])
        graph.add_edge(movie['Name'], movie['Description'])
        graph.add_edge(movie['Name'], movie['Genre'])

        # find the top top_n most similar movies based on genre
        cosine_similarities = cosine_similarity(tf[i:i + 1], tf).flatten()
        related_docs_indices = sorted(range(len(cosine_similarities)), key=cosine_similarities.__getitem__,
                                      reverse=True)
        similar_movies = [idx for idx in related_docs_indices if idx != i][:top_n]

        for j in similar_movies:
            graph.add_edge(movie['Name'], movies['Name'].loc[j], label = 'similar')

    return graph


def adamic_adar_index(graph, movie_seen, recommended_movie):
    """
    Calculates the Adamic-Adar index for two movies in a graph.

    The Adamic-Adar index measures the similarity between two nodes based on their
    shared neighbors. It assigns higher weight to neighbors with fewer connections
    (less common nodes), emphasizing unique relationships.

    Formula:
        Adamic-Adar score = Σ (1 / log(degree(neighbor)))

    Where:
        - `neighbor` is a common neighbor of `movie_seen` and `recommended_movie`.
        - `degree(neighbor)` is the number of edges connected to the neighbor.
    """
    # Get the neighbors of both movies
    common_neighbors = set(graph.neighbors(movie_seen)).intersection(graph.neighbors(recommended_movie))

    # Calculate the Adamic-Adar weight based on common neighbors
    return sum(1 / math.log(graph.degree(neighbor)) for neighbor in common_neighbors if graph.degree(neighbor) > 1)


def get_recommendations(graph, movie_seen, top_n=5):
    """
    Generates movie recommendations based on the Adamic-Adar index using a network graph.

    This function identifies movies related to the given movie (`movie_seen`) by analyzing
    the graph structure. It calculates the Adamic-Adar index for second-degree neighbors
    (movies connected through shared nodes) and ranks them to recommend the top `n` movies.

    1. Traverses the graph to identify second-degree neighbors of the `movie_seen` node.
    2. Filters nodes to include only movies that have not already been recommended.
    3. Computes the Adamic-Adar index to measure similarity between `movie_seen` and candidate movies.
    4. Returns the top `n` recommendations based on the highest scores.

    return A sorted list of recommended movies with their Adamic-Adar scores.
    """
    recommendations = []
    already_seen_movies = set()

    # Loop through neighbors of the movie seen
    for neighbor_inner in graph.neighbors(movie_seen):
        # Loop through neighbors of neighbors (second-degree neighbors)
        for neighbor_outer in graph.neighbors(neighbor_inner):
            # Find common neighbors now for the adamic adar index
            if neighbor_outer == movie_seen:
                continue  # Skip if the node is the movie seen:

            # Check if it's a movie node and not already visited
            if 'label' in graph.nodes[neighbor_outer] and graph.nodes[neighbor_outer]['label'] == "Movie":
                if neighbor_outer not in already_seen_movies:
                    # Calculate Adamic-Adar score and add to recommendations
                    score = adamic_adar_index(graph, movie_seen, neighbor_outer)
                    recommendations.append((neighbor_outer, score))

                    # add to keep track movie has been seen
                    already_seen_movies.add(neighbor_outer)

    # Sort by score and return top N recommendations
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def tokenize_with_context(column):
    """
    Preprocesses and lemmatizes a column of text data.

       - Tokenizes each document in the input column.
       - Tags tokens with parts of speech (POS) and converts tags to WordNet format.
       - Lemmatizes tokens based on their POS tags.
       - Removes non-alphabetic characters from tokens.
       - Filters out stop words and empty tokens.

    Returns a list of preprocessed, lemmatized, and filtered text documents as strings.
    """

    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    data_col = []
    for index, doc in enumerate(column):

        pos_tags = nltk.pos_tag(nltk.word_tokenize(doc))
        wordnet_tags = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tags))

        lemmatized_tokens = []
        for word, tag in wordnet_tags:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_tokens.append(str(word).lower())
            else:
                # else use the tag to lemmatize the token
                lemma = lemmatizer.lemmatize(word, tag)
                lemmatized_tokens.append(lemma.lower())

        tokens_removed_char = [re.sub(r'[^a-zA-Z]', '', w) for w in lemmatized_tokens]
        final_tokens = [word for word in tokens_removed_char if word and word not in stop_words]

        joined_tokens = " ".join(final_tokens)
        data_col.append(joined_tokens)

    return data_col

# Main method call this function to create graph and get top 5 recommendation
def network_recommendation(org_movie, n = 105):
    """
    Creates a network-based movie recommendation system by clustering and graph analysis.

    Steps:
    1. **Top Movies Selection:**
       - Selects the top `n` movies using a Bayesian weighted average of ratings and votes.
    2. **Feature Vectorization:**
       - Applies TF-IDF vectorization to movie summart plot, genres, and combined directors/stars
         for capturing textual similarity.
    3. **Clustering:**
       - Uses K-means clustering to group movies based on summary plot, genres, and directors/stars.
       - Assigns each movie to a cluster.
    4. **Visualization of Clusters:**
       - Displays the distribution of movies across clusters for analysis.
    5. **Network Graph Creation:**
       - Builds a graph where nodes represent movies and edges connect movies based on clustering
         and similarity.
    6. **Recommendation with Adamic-Adar Index:**
       - Recommends the most similar movies to a given input movie using the Adamic-Adar index.
    7. **Visualization of Recommendations:**
       - Displays recommended movies and visualizes the network graph, highlighting recommendation
         paths and nodes.
    """


    # To not accidentally overwrite the dataset
    movies = org_movie.copy()

    # To pick the top n movies from the dataset we use
    # Bayesian weighted average on rating_value and votes combined
    top_movies = Bayesian_weighted_average(movies, n)

    # Create TF-IDF on summary plot, Genre and combined Directors and Stars
    # Reusing the tokenize_with_context from earlier
    text_content = tokenize_with_context(top_movies['Description'])
    text_genre = tokenize_with_context(top_movies['Genre'])
    text_Stars = tokenize_with_context(top_movies['Directors'] + top_movies['stars'])

    vector = TfidfVectorizer()
    vector_ge = TfidfVectorizer()
    vector_stars = TfidfVectorizer()

    tf = vector.fit_transform(text_content)
    tf_genre = vector_ge.fit_transform(text_genre)
    tf_stars = vector_stars.fit_transform(text_Stars)

    # Create the clusters using K-means on Summary plot, Genre and combined Directors and stars
    # clusters for summary plot
    kmean = KMeans(n_clusters = round(n / 10), random_state=0, n_init="auto")
    kmean.fit(tf)

    # Cluster for genre
    kmean_ge = KMeans(n_clusters= 19, random_state=0, n_init="auto")
    kmean_ge.fit(tf_genre)

    # Cluster for stars
    kmean_st = KMeans(n_clusters= round(n / 3), random_state=0, n_init="auto")
    kmean_st.fit(tf_stars)

    # convert summary plot, genre and combined directors and stars to vector
    des_vec = vector.transform(top_movies['Description'])
    ge_vec = vector_ge.transform(top_movies['Genre'])
    st_vec = vector_stars.transform(top_movies['Directors'] + top_movies['stars'])

    # Assign movie to cluster
    c = kmean.predict(des_vec)
    k = kmean_ge.predict(ge_vec)
    st = kmean_st.predict(st_vec)

    # Replace the summary plot, Genre and combined Director and stars
    # with their respective assigned cluster in the dataframe of top movies
    top_movies['Description'] = c
    top_movies['Genre'] = k
    top_movies['Directors'] = st

    # Visualization of the movie distribution on how the movies
    # the movies got assigned in clustering
    show_distrubtion(top_movies)

    # Create the movie graph drawing connection between the movies
    # using the clustered Summary plot, Genre, combined directors and stars
    # and similarity to the top 5 most similar movies based of the genre clustering
    graph = create_graph(top_movies, tf_genre)

    # Display the network graph to see how the movies are connected
    #plot_graph(graph)

    # To get recommendation from the graph using the adamic adar index
    # top_n = 5
    movie_seen = "Toy Story"
    recommended_movies = get_recommendations(graph, movie_seen, top_n=5)
    print(recommended_movies)

    # Plot the recommended movies based on the already "seen movie"
    # displaying highest scoring adamic adar index and movies
    plot_simpel_recommended_movies(recommended_movies, movie_seen)

    # Display the Network graph where the seen movie is highlighted green
    # and recommended movies yellow and their paths
    plot_graph_n_recommended(graph, recommended_movies, movie_seen)