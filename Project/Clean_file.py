import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import regex as re

def check_nltk_resource(resource_name):
    try:
        # Check if the resource is available
        nltk.data.find(f'corpora/{resource_name}')
        print(f"{resource_name} is already downloaded.")
    except LookupError:
        # Resource is not available, download it
        print(f"{resource_name} not found. Downloading now...")
        nltk.download(resource_name)

def clean_data(movies):
    """
    Cleans the 'Description' field in the given movie dataset.

    This function performs several preprocessing steps to standardize and clean the textual descriptions of movies. These steps include:
    - Converting text to lowercase.
    - Splitting text into individual words (tokenization).
    - Removing special characters from words (e.g., punctuation or numbers).
    - Removing English stopwords (common words like "the", "and", etc.).
    - Lemmatizing words to reduce them to their base forms (e.g., "running" becomes "run").

    Parameters:
        movies (pd.DataFrame): A pandas DataFrame containing a column named 'Description', which holds the text descriptions of movies.

    Returns:
        Currently None
        Idea:
        The function modifies the DataFrame in-place, cleaning the 'Description' field.
    """

    # Needed data to clean with:
    check_nltk_resource('stopwords')
    check_nltk_resource('wordnet')

    # Description clean
    #des = movies['Description'][0]
    i = 0
    while i < len(movies):
        des = movies['Description'][i]

        # Lower case
        des = des.lower()

        # Split it
        des = des.split()

        # Remove special characters
        des = [re.sub(r'[^a-zA-Z]', '', w) for w in des]

        # Remove stop words
        stop_words = set(stopwords.words('english'))

        # Filtered words
        filter_w = []
        for w in des:
            if w not in stop_words:
                filter_w.append(w)

        des = filter_w

        # Lemmatization of words in description
        lemmatizer = WordNetLemmatizer()
        des = [lemmatizer.lemmatize(w) for w in des]

        # Last clean up
        des = [token for token in des if token]


        print(des)
        i += 1

def check_for_duplicates(movies):
    """
    Checks dataframe of movies for if any dublicates occur by checking the movie ID
    """
    # Check for duplicate ID in movies list
    duplicates = movies[movies['ID'].duplicated(keep=False)]  # Keep all duplicates (both original and repeated)

    # Display duplicates
    if not duplicates.empty:
        print("Duplicate Names Found:")
        print(duplicates)
    else:
        print("No duplicate names found.")