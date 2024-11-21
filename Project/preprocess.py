import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import regex as re
import pandas as pd

def check_nltk_resource(resource_name):
    """
    Just checks if necessary resources like stopwords and wordnet is downloaded
    """
    try:
        # Check if the resource is available
        nltk.data.find(f'corpora/{resource_name}')
        print(f"{resource_name} is already downloaded.")
    except LookupError:
        # Resource is not available, download it
        print(f"{resource_name} not found. Downloading now...")
        nltk.download(resource_name)

def check_for_duplicates(movies):
    """
    Simple method to validate that the dataframe has no doublicates
    """
    # Check for duplicate ID in movies list
    duplicates = movies[movies['ID'].duplicated(keep=False)]  # Keep all duplicates (both original and repeated)

    # Display duplicates
    if not duplicates.empty:
        print("Duplicate Names Found:")
        print(duplicates)
    else:
        print("No duplicate names found.")

def tokenize_movie_descriptions(org_movies):
    """
       Cleans and preprocesses the movie descriptions in the given dataset.

       This function performs the following steps:
       1. Converts descriptions to lowercase.
       2. Tokenizes the descriptions into words.
       3. Removes special characters and non-alphabetic tokens.
       4. Removes stop words using the NLTK stopword list.
       5. Lemmatizes the remaining words to their base form.
       6. Stores the cleaned, tokenized descriptions back in the dataset.

       Args:
           movies (pd.DataFrame): A DataFrame containing a column 'Description' with movie descriptions.

       Returns:
           None: Modifies the input DataFrame in-place by replacing 'Description' with a list of cleaned tokens.
       """

    # Currently we don't want to change the original dataframe so:
    # Copy as to not change the original list of movies
    movies = org_movies.copy()

    # Needed data to clean with:
    check_nltk_resource('stopwords')
    check_nltk_resource('wordnet')

    # Initialize stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Process descriptions
    for i in range(len(movies)):
        des = movies['Description'][i]

        # Lower case
        des = des.lower()

        # Split it
        des = des.split()

        # Remove special characters
        des = [re.sub(r'[^a-zA-Z]', '', w) for w in des]

        # Remove stop words
        des = [word for word in des if word and word not in stop_words]

        # Lemmatization of words in description
        des = [lemmatizer.lemmatize(w) for w in des]

        # Last clean up
        des = [token for token in des if token]

        # Update the dataset with tokenized descriptions
        movies['Description'][i] = des

    return movies

def split_rating(org_movies):
    """
    This method splits the rating into two:
    Rating_value (rating of movie by user)
    votes (votes for how many users have rated the movie)

    Rows are dropped if movies with votes < 1000 are considered not popular enough
    """
    # Create a copy of the DataFrame
    movies = org_movies.copy()

    # Remove rows where 'Rating' is 'N/A'
    movies = movies[movies['Rating'] != 'N/A']

    # Extract rating value and votes using regex
    pattern = r'([0-9.]+) \(([\d.]+)([KM])\)'
    matches = movies['Rating'].str.extract(pattern)

    # Convert rating value to float
    movies['Rating_value'] = matches[0].astype(float)

    # Process votes: Convert 'K' to 1,000 and 'M' to 1,000,000
    votes = matches[1].astype(float)
    multiplier = matches[2].map({'K': 1_000, 'M': 1_000_000})
    movies['votes'] = votes * multiplier

    # Keep only rows where 'Rating' matched the regex
    movies = movies.dropna(subset=['Rating_value', 'votes'])

    # Reset the index
    movies.reset_index(drop=True, inplace=True)

    return movies


def remove_series(org_movies):
    """
    This method removes series from the dataset.
    Series are identified by a year range like "2022-2020",
    """
    # Not to overwrite original data
    movies = org_movies.copy()

    indices_to_remove = []

    # Process ratings: Loop through the DataFrame and identify rows to remove
    for i in range(len(movies)):
        year = movies['Year'][i]
        if year.isnumeric() == False:
            indices_to_remove.append(i)

    # Remove the identified rows
    movies.drop(indices_to_remove, inplace=True)

    # Reset index after filtering
    movies.reset_index(drop=True, inplace=True)

    return movies

def remove_NA(org_movies):
    """
    Remove N/A indexes if they occur in:
    Year, Length, Age or Rating
    """
    # Make a copy of the original DataFrame to avoid overwriting it
    movies = org_movies.copy()

    # Define columns to check
    columns_to_check = ["Year", "Length", "Age", "Rating"]

    indices_to_remove = []

    print(movies["Name"][232] + " " + str(movies["Age"][232]))

    for i in range(len(movies)):
        for c in columns_to_check:
            if str(movies[c][i]) == "nan":
                indices_to_remove.append(i)

    # Remove the identified rows
    movies.drop(indices_to_remove, inplace=True)

    # Reset index after filtering
    movies.reset_index(drop=True, inplace=True)

    return movies


def clean_tits(org_movies):
    """
    Method to remove indexes which contain N/A, series and split rating into:
    Rating_value - overall rating based on the votes of the users
    votes - number of votes by the user
    """
    movies = remove_NA(org_movies)
    movies = split_rating(movies)
    movies = remove_series(movies)
    return movies