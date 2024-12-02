from selenium import webdriver
import regex as re
from concurrent.futures import ThreadPoolExecutor
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def get_movie_detail(details, num_2):
    """
        Extracts movie metadata (year, length, and age rating) from a list of detail elements.

        This helper function parses and validates specific pieces of information from the `details` list:
        - **Year:** Matches a single year (e.g., "2023") or a year range (e.g., "2010–2022").
        - **Length:** Matches runtime in formats like "2h 30m", "1h", "45m", or episode counts like "8 eps".
        - **Age Rating:** Matches common age classifications such as "PG-13", "18", "TV-MA", or equivalent terms ("Not Rated", "Approved", etc.).

        The function iterates through the `details` list starting from index `num_2` and extracts metadata based on regex patterns. It ensures valid values are captured and updates the index for subsequent parsing.

        Parameters:
            details (list): A list of elements (e.g., Selenium `WebElement`s) containing movie metadata as text.
            num_2 (int): The current index in the `details` list from which to start parsing.

        Returns:
            tuple: A tuple containing:
                - `year` (str): The extracted year or "N/A" if not found.
                - `length` (str): The extracted runtime or "N/A" if not found.
                - `age` (str): The extracted age rating or "N/A" if not found.
                - `num_2` (int): The updated index after parsing the metadata.

        Notes:
        - The function uses default values ("N/A") if a particular piece of metadata is not found or doesn't match the expected format.
        - Supports a variety of age rating formats across different regions and systems.
        """
    # Expressions for validation
    year_pattern = re.compile(r"^\d{4}$|^\d{4}–\d{4}$")  # Matches a single year or a year range (e.g., 2010 or 2010–2022)
    length_pattern = re.compile(r"^\d+h \d+m$|^\d+h$|^\d+m$|^\d+ eps$")  # Matches "Xh Ym" format or "X eps"
    age_pattern = re.compile(
        r"^\d{1,2}$|^PG-?\d{0,2}$|^X$|^12A$|^U$|^18$|^NR$|^R$|^AA$|^G$|^NC-17$|^M$|^MA$|^T$|^L$|^K-7$|^K-12$|^A$|^S$|^TV-MA$|^TV-PG$|^TV-Y7-FV$"
    )  # Age rating can be like "18", "PG", "PG-13", "12A", "X", "U" and "TV-MA"

    # Default values
    year, length, age = "N/A", "N/A", "N/A"


    # Check year
    if num_2 < len(details) and year_pattern.match(details[num_2].text):
        year = details[num_2].text
        num_2 += 1
    elif  num_2 < len(details) and year_pattern.match(details[num_2 + 1].text):
        year = details[num_2].text
        num_2 += 2

    # Check length
    if num_2 < len(details) and length_pattern.match(details[num_2].text):
        length = details[num_2].text
        num_2 += 1

    # Check age
    if num_2 < len(details) and (age_pattern.match(details[num_2].text) or details[num_2].text in {"Not Rated", "Rejected", "Approved", "Unrated"}):
        age = details[num_2].text
        num_2 += 1

    return year, length, age, num_2


def fetch_titles(target_movie = 10000):
    # Save movie and ID in dictionary
    titles = {}

    # Links to different IMDB movie lits mostly seperated by genre
    urls = ['https://www.imdb.com/list/ls000634294/', 'https://www.imdb.com/list/ls050782187/',
            'https://www.imdb.com/list/ls059633855/', 'https://www.imdb.com/list/ls074612774/',
            'https://www.imdb.com/list/ls066207865/', 'https://www.imdb.com/list/ls092264063/',
            'https://www.imdb.com/list/ls058416162/', 'https://www.imdb.com/list/ls063897780/',
            'https://www.imdb.com/list/ls058726648/', 'https://www.imdb.com/list/ls000712763/',
            'https://www.imdb.com/list/ls072723591/', 'https://www.imdb.com/list/ls072723351/',
            'https://www.imdb.com/list/ls063361223/', 'https://www.imdb.com/list/ls044085335/',
            'https://www.imdb.com/list/ls044085335/', 'https://www.imdb.com/list/ls025833831/',
            'https://www.imdb.com/list/ls066980750/', 'https://www.imdb.com/list/ls072723203/',
            'https://www.imdb.com/list/ls062101934/', 'https://www.imdb.com/list/ls063397905/',
            'https://www.imdb.com/list/ls062913334/', 'https://www.imdb.com/list/ls062152451/',
            'https://www.imdb.com/list/ls027351811/', 'https://www.imdb.com/list/ls057336010/']

    # Webscrape one site at a time:
    for url in urls:
        web = webdriver.Chrome()
        web.get(url)

        # Load website
        time.sleep(3)

        # Close pop up
        web.find_element("xpath", "/html/body/div[2]/div/div/div[2]/div/button[1]").click()

        # Regex pattern to get the title, year, movie length, age restriction
        title_pattern = r"^\d+\.\s*(.*)"

        # Make the bot scroll to the buttom of the page for 100*0.2 sec
        i = 0
        while True:
            web.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.2)  # Wait for more titles to load
            print(i)
            i += 1
            if i > 100:
                break

        # Pull all titles from the list
        cur_titles = web.find_elements('xpath', "//h3[@class='ipc-title__text']")

        # Pull all descriptions etc
        des = web.find_elements('class name', "ipc-html-content-inner-div")
        details = web.find_elements('xpath', "//span[contains(@class, 'dli-title-metadata-item')]")
        ratings = web.find_elements('class name', "ipc-rating-star--rating")
        vote_counts = web.find_elements('class name', "ipc-rating-star--voteCount")

        # To index
        num = 1
        num_2 = 0

        for title in cur_titles:
            # This activates when reached end of list
            if(title.text == "More to explore"):
                break

            #movie_name = re.search(title_pattern, title.text).group(1)
            match = re.search(title_pattern, title.text)

            if match:
                movie_name = match.group(1)  # Extract movie name
            else:
                movie_name = "Unknown Title"  # Default to a fallback title
                print(f"Warning: No match found for title text: {title.text}")  # Debugging line

            link = web.find_element('partial link text', title.text).get_attribute("href")

            # Pull the ID out:
            movie_id = link.split('/title/')[1].split('/')[0]

            # Map based on ID
            if titles.get(movie_id) is None:
                # Get year, length, age, and updated num_2 index
                year, length, age, num_2 = get_movie_detail(details, num_2)

                # Map the movie details to the dictionary
                titles[movie_id] = {
                    'title': movie_name,
                    'description': des[num].text,
                    'year': year,
                    'length': length,
                    'age': age,
                    'rating': ratings[num - 1].text + vote_counts[num - 1].text
                }
            else:
                # just a dummy function to have the counter follow
                year, length, age, num_2 = get_movie_detail(details, num_2)

            # Move onto next movie title
            num += 1

            # If we wish to early stop
            if len(titles) > target_movie:
                break

        web.quit()


    # Convert from dictionary to pandas list:
    # first from dictionary to a list of tuples [(Title, ID), (Title, ID), ...]
    movies_list = [
        (details['title'], movie_id, details['year'], details['length'], details['age'], details['rating'], details['description'])
        for movie_id, details in titles.items()
    ]

    # Convert the list to a pandas DataFrame, with 'Name' first and 'ID' second
    df_movies = pd.DataFrame(movies_list, columns=['Name', 'ID', 'Year', 'Length', 'Age', 'Rating', 'Description'])

    # Set pandas display option to print all rows
    pd.set_option('display.max_rows', None)  # None means no limit

    # Print the entire DataFrame
    print(df_movies.loc[:, df_movies.columns != 'Description'])

    # Save it as csv file
    df_movies.to_csv(f'movie_titles_and_ids.csv', index=False)


def fetch_movie_details(movie):
    """
    Scrapes movie details from multiple IMDb lists and compiles the data into a pandas DataFrame.

    * Code ran for 1613 sec to get 3123 titles *

    This function automates the process of:
    - Visiting IMDb movie list pages.
    - Extracting movie details such as title, description, year, length, age restriction, and rating.
    - Storing the collected data in a dictionary mapped by IMDb movie IDs.
    - Saving the final data as a CSV file for further use.

    Key Steps:
    1. **Scraping Titles and Metadata:** The function navigates through IMDb list URLs, scrolling down the pages to load all movie entries.
    2. **Extracting Data:** Using regex and XPath, it extracts:
        - Title
        - Description
        - Year
        - Length
        - Age restriction
        - IMDb rating and vote count
    3. **Avoiding Duplicate Entries:** Ensures movies with the same IMDb ID are not duplicated.
    4. **Early Stopping:** Stops the scraping process once the target number of movies (`target_movie`) is reached.
    5. **Data Organization:** Converts the collected data into a pandas DataFrame and saves it to a CSV file.

    Parameters:
        target_movie (int): The number of movies to scrape before stopping (default is 2000).

    Returns:
        None: Saves the resulting dataset as a CSV file named 'movie_titles_and_ids.csv'.

    Notes:
    - The function handles pop-ups and dynamically loads additional content by scrolling.
    - Requires the `get_movie_detail()` helper function for extracting year, length, and age restriction from metadata.
    - Uses Selenium with Chrome WebDriver; ensure the WebDriver is installed and configured properly.
    """

    # Movie title (ID)
    link = "https://www.imdb.com/title/" + movie
    web_address = link
    web = webdriver.Chrome()
    web.get(web_address)

    # Initialize lists for genres, directors, and stars
    genres, directors, stars = [], [], []

    try:
        # Wait for genre elements to be present
        WebDriverWait(web, 20).until(
            EC.presence_of_all_elements_located(('xpath', "//a[@class='ipc-chip ipc-chip--on-baseAlt']/span"))
        )

        # Locate all genre elements using XPath
        genre_elements = web.find_elements('xpath', "//a[@class='ipc-chip ipc-chip--on-baseAlt']/span")
        genres = [genre_element.text for genre_element in genre_elements]

    except Exception as e:
        print(f"Error extracting genre for {movie}: {e}")

    try:
        # Wait for director elements to be present
        WebDriverWait(web, 20).until(
            EC.presence_of_all_elements_located(
                ('xpath', "//li[@data-testid='title-pc-principal-credit']//a"))
        )

        # Locate all director elements using XPath
        director_elements = web.find_elements('xpath', "//li[@data-testid='title-pc-principal-credit']//a")
        directors = [director.text for director in director_elements if "ref_=tt_ov_dr" in director.get_attribute("href")]

    except Exception as e:
        print(f"Error extracting directors for {movie}: {e}")

    try:
        # Extract stars
        WebDriverWait(web, 20).until(
            EC.presence_of_all_elements_located(('xpath', "//li[@data-testid='title-pc-principal-credit']//a"))
        )
        star_elements = web.find_elements('xpath', "//li[@data-testid='title-pc-principal-credit']//a")
        stars = [
            star.text for star in star_elements
            if "ref_=tt_ov_st" in star.get_attribute("href")  # Ensure it's a "star" link
        ]

        # Clean up stars: remove "Stars" and empty strings
        stars = [star for star in stars if star and star != "Stars"]

    except Exception as e:
        print(f"Error extracting stars for {movie}: {e}")

    # Close the browser
    web.quit()

    return genres, directors, stars

def safe_fetch_movie_details(movie_id):
    try:
        return fetch_movie_details(movie_id)
    except Exception as e:
        print(f"Error processing ID {movie_id}: {e}")
        return None  # Return None or any placeholder to indicate failure

def get_genre_n_production(movies):
    """
    * Code functions but could use a clean up *

    * Ran for 4783 sec for 3064 titles *

    Scrapes additional details (genres, directors, and stars) for a given movie from its IMDb page.

    This function automates the process of visiting an IMDb movie page using its ID, extracting:
    - **Genres:** A list of genres associated with the movie.
    - **Directors:** A list of directors, filtered to include only relevant links.
    - **Stars:** A list of stars (cast), cleaned to exclude irrelevant entries.

    Process:
    - Opens the movie's IMDb page using Selenium's Chrome WebDriver.
    - Waits for the necessary elements (genres, directors, and stars) to load using explicit waits.
    - Extracts text data from specific HTML elements identified by their XPath.
    - Handles errors gracefully if any elements are not found or time out.
    - Closes the browser after scraping.

    Parameters:
        movie (str): The IMDb ID of the movie (e.g., "tt1234567").

    Returns:
        tuple: A tuple containing three lists:
            - `genres` (list): The genres of the movie.
            - `directors` (list): The directors of the movie.
            - `stars` (list): The stars (cast) of the movie.
    """
    # Creating a copy of the original DataFrame to avoid modifying it directly
    movies_to_process = movies.copy()

    # ThreadPoolExecutor for parallel processing to webscape 10 websites at a time
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Fetch genres for all the movies
        try:
            results = list(executor.map(safe_fetch_movie_details, movies_to_process['ID']))
        except:
            print("error")

    # Extract genres, directors, and stars from the results
    genres = [result[0] for result in results]

    # Remove empty strings from the directors list by filtering out unwanted empty values
    directors = [
        [director for director in result[1] if director.strip()]  # Remove empty strings or spaces
        for result in results
    ]

    # Clean the list in case "Directors" or "Director" accidentally is included
    directors = [
        [entry for entry in directors_list if entry not in {"Director", "Directors"}]
        if len(directors_list) > 0 else []
        for directors_list in directors
    ]

    # Extract stars (if necessary) in a similar fashion
    stars = [result[2] for result in results]

    # Add the genres to the DataFrame as new columns
    movies_to_process['Genre'] = genres
    movies_to_process['Directors'] = directors
    movies_to_process['stars'] = stars

    # Save the updated DataFrame to a new CSV file
    movies_to_process.to_csv('movies_with_genres.csv', index=False)