import preprocess as pp
from collections import Counter
import math
import numpy as np

def tfidf_values(org_movies):
    """
    This method computes the TF-IDF (Term Frequency-Inverse Document Frequency)
    scores for a collection of movie descriptions, transforming them into a
    numerical representation useful for tasks like clustering or recommendation systems.
    """
    # Get a new list with the tokenized description
    movies = pp.tokenize_movie_descriptions(org_movies)

    # Create a vocabulary of the tokenized
    vocabulary = create_vocabulary(movies)

    # Compute the term frequency
    tf_values = term_frequency(movies, vocabulary)

    # Compute the inverse document frequency (IDF) values
    idf_values = inverse_document_freq(movies, vocabulary)

    # Compute the TF-IDF values by multiplying TF and IDF
    tfidf_scores = compute_tfidf(tf_values, idf_values)

    # Convert tfidf_scores (list of dictionaries) to a 2D array
    # ? Not sure about this ?
    tfidf_scores_2d = np.array([list(tfidf.values()) for tfidf in tfidf_scores])

    return tfidf_scores_2d


def create_vocabulary(movies):
    """
    Create vocabulary used for TF IDF
    """
    # Set to store all unique words across the different titles description
    vocabulary = set()

    for i in range(len(movies)):
        # Add token to vocabulary
        tokens = movies['Description'][i]
        vocabulary.update(tokens)

    # Return the set (sorted in this case)
    return sorted(vocabulary)

def term_frequency(movies, vocabulary):
    """
    Computes Term Frequency (TF) for movie descriptions.
    """
    term_frequency = []

    for description in movies['Description']:
        # Count occurrences of each word in the description
        term_count = Counter(description)

        # Total number of terms in the description
        total_terms = len(description)

        # Compute TF for each term in the vocabulary
        tf_doc = {term: term_count[term] / total_terms if total_terms > 0 else 0 for term in vocabulary}

        # Append TF dictionary for this description
        term_frequency.append(tf_doc)

    return term_frequency

def inverse_document_freq(movies, vocabulary):
    """
    Computes Inverse Document Frequency (IDF) for terms in the vocabulary.
    """
    idf_values = {}

    # Total number of documents (movies)
    total_documents = len(movies)

    for term in vocabulary:
        # Count the number of documents that contain the term
        doc_count = sum(1 for description in movies['Description'] if term in description)

        # Compute IDF using the formula: log(N / df)
        idf_values[term] = math.log(total_documents / (doc_count + 1))  # Adding 1 to avoid division by 0

    return idf_values

def compute_tfidf(tf_values, idf_values):
    """
    Computes the TF-IDF values by multiplying TF and IDF.
    """
    tfidf_scores = []

    for tf_doc in tf_values:
        tfidf_doc = {term: tf_doc[term] * idf_values[term] for term in tf_doc}
        tfidf_scores.append(tfidf_doc)

    return tfidf_scores