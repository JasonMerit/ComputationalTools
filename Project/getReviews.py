import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from tqdm import tqdm
import pickle

headers = {'user-agent': 'my-app/0.0.1', 'Accept-Language': 'en-US,en;q=0.5'}

def gen_dict_extract(key, var):
    if hasattr(var,'items'): # hasattr(var,'items') for python 3
        for k, v in var.items(): # var.items() for python 3
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result

def scrapeReviews(ImdbId):
    
    movie_url = f'https://www.imdb.com/title/{ImdbId}/reviews/'
    # print(movie_url)
    r = requests.get(url=movie_url, headers=headers)
    if r.status_code != 200:
        raise Exception("Error in response. Bad URL?")
    soup = BeautifulSoup(r.text, 'html.parser')
    # print(soup.prettify())

    # Find all <script id="__NEXT_DATA__" type="application/json">
    json_data2 = soup.find('script', id='__NEXT_DATA__').text
    json_data2 = json.loads(json_data2)

    reviews = list(gen_dict_extract('plaidHtml', json_data2))
    if len(reviews) != 25:
        return None
    
    return reviews




if __name__ == '__main__':
    with open('Project/data/reviews.pkl', 'rb') as f:
        reviews = pickle.load(f)
    titles = pd.read_csv('Project/movie_titles_and_ids.csv')
    for id in tqdm(titles['ID']):
        if id in reviews:
            continue
        reviews[id] = scrapeReviews(id)
        pickle.dump(reviews, open('Project/data/reviews.pkl', 'wb'))
