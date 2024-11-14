from selenium import webdriver
import time
import regex as re
import pandas as pd

def get_movie_detail(details, num_2):
    # Expressions for validation
    year_pattern = re.compile(r"^\d{4}$|^\d{4}–\d{4}$")  # Matches a single year or a year range (e.g., 2010 or 2010–2022)
    length_pattern = re.compile(r"^\d+h \d+m$|^\d+h$|^\d+m$|^\d+ eps$")  # Matches "Xh Ym" format or "X eps"
    age_pattern = re.compile(r"^\d{1,2}$|^PG-?\d{0,2}$|^X$|^12A$|^U$|^18$|^NR$|^R$|^AA$|^G$|^NC-17$|^M$|^MA$|^T$|^L$|^K-7$|^K-12$|^A$|^S$")  # Age rating can be like "18", "PG", "PG-13", "12A", "X" and "U"

    # Default values
    year, length, age = "N/A", "N/A", "N/A"

    # Check year
    if num_2 < len(details) and year_pattern.match(details[num_2].text):
        year = details[num_2].text
        num_2 += 1

    # Check length
    if num_2 < len(details) and length_pattern.match(details[num_2].text):
        length = details[num_2].text
        num_2 += 1

    # Check age
    if num_2 < len(details) and (age_pattern.match(details[num_2].text) or details[num_2].text == "Not Rated"):
        age = details[num_2].text
        num_2 += 1

    return year, length, age, num_2

def fetch_movie_titles_and_ID(target_movie = 1000):
    # Target movie to be extracted
    #target_movie = 1000

    # Save movie and ID in dictionary
    titles = {}

    # Website of "Full Movie List" IMDB
    web_address = "https://www.imdb.com/list/ls000634294/"
    web = webdriver.Chrome()
    web.get(web_address)

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
        if (title.text == "More to explore"):
            break

        # Find the hyperlink by movie name
        movie_name = re.search(title_pattern, title.text).group(1)
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

    # Convert from dictionary to pandas list:
    # first from dictionary to a list of tuples [(Title, ID), (Title, ID), ...]
    movies_list = [
        (details['title'], movie_id, details['year'], details['length'], details['age'], details['rating'],
         details['description'])
        for movie_id, details in titles.items()
    ]

    # Convert the list to a pandas DataFrame, with 'Name' first and 'ID' second
    df_movies = pd.DataFrame(movies_list, columns=['Name', 'ID', 'Year', 'Length', 'Age', 'Rating', 'Description'])

    # Set pandas display option to print all rows
    pd.set_option('display.max_rows', None)  # None means no limit

    # Print the entire DataFrame
    print(df_movies.loc[:, df_movies.columns != 'Description'])

    # Save it as csv file
    df_movies.to_csv('movie_titles_and_ids.csv', index=False)

    return df_movies