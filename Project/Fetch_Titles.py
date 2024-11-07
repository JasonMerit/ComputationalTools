from selenium import webdriver
import time
import regex as re
import pandas as pd

def fetch_movie_titles_and_ID(num = 1000):
    # Target movie to be extracted
    target_movie = num

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

    # Regex pattern to get only the title
    pattern = r"^\d+\.\s*(.*)"
    previous = -1
    while len(titles) < target_movie:
        # Pull all titles from the list
        cur_titles = web.find_elements('xpath', "//h3[@class='ipc-title__text']")

        # Go through first 25 movies
        for title in cur_titles:
            # This activates when reached end of list
            if (title.text == "More to explore"):
                break

            # Find the hyperlink by movie name
            movie_name = re.search(pattern, title.text).group(1)
            link = web.find_element('partial link text', title.text).get_attribute("href")

            # Pull the ID out:
            movie_id = link.split('/title/')[1].split('/')[0]

            # Map based on ID
            if titles.get(movie_id) is None:
                titles[movie_id] = movie_name

        # Incase it gets stuck
        if previous == len(titles):
            break

        previous = len(titles)

        # Scroll down to load more titles
        web.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for more titles to load

    # Convert from dictionary to pandas list:
    # first from dictionary to a list of tuples [(Title, ID), (Title, ID), ...]
    movies_list = [(title, movie_id) for movie_id, title in titles.items()]

    # Convert the list to a pandas DataFrame, with 'Name' first and 'ID' second
    df_movies = pd.DataFrame(movies_list, columns=['Name', 'ID'])

    # Set pandas display option to print all rows
    #pd.set_option('display.max_rows', None)  # None means no limit

    # Print the entire DataFrame
    #print(df_movies)

    # Save it as csv file
    #df_movies.to_csv('movie_titles_and_ids.csv', index=False)

    # Return dataframe
    return df_movies