import requests
from bs4 import BeautifulSoup
import json
import time
import csv

headers = {'user-agent': 'my-app/0.0.1', 'Accept-Language': 'en-US,en;q=0.5'}

def get_data(id):
    movie_url = f'https://www.imdb.com/title/{id}/'
    r = requests.get(url=movie_url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    

    json_data1 = soup.find('script', type='application/ld+json').text
    json_data1 = json.loads(json_data1)


    aggregateRating = json_data1['aggregateRating']
    release_date = json_data1['datePublished']
    actor = json_data1['actor']
    creator = json_data1['creator']
    director = json_data1['director']
    json_data2 = json.loads(soup.find('script', id='__NEXT_DATA__').text)

    return [
        json_data1['name'],
        json_data1['url'].split('/')[4],
        json_data1['description'],
        str(aggregateRating['ratingValue']),
        str(aggregateRating['ratingCount']),
        json_data1['contentRating'],
        "/".join(json_data1['genre']),
        release_date.split('-')[0],
        json_data1['keywords'],
        "/".join([a['name'] for a in actor]),
        "/".join([c['name'] for c in creator if c['@type'] == 'Person']),
        "/".join([d['name'] for d in director]),
        json_data1['duration'],
        str(json_data2['props']['pageProps']['aboveTheFoldData']['metacritic']['metascore']['score']),
    ]



def main(debug):
    t0 = time.time()
    if not debug:
        ids = ['tt0111161','tt0068646','tt0110912','tt0071562','tt1375666','tt0167260','tt0076759','tt0120737','tt0133093','tt0114369','tt0211915','tt0103064','tt1520211','tt1156398','tt0365748','tt0480249','tt0455407','tt0462322','tt0289043','tt0463854','tt0432021','tt0363547','tt0120804', 'tt1077258']
        print(len(ids))
        quit()
        for id in ids:
            try:
                row = get_data(id)
                with open('Project/data/movies_metrics.csv','a', newline='') as fd:
                    writer = csv.writer(fd, delimiter=chr(255))
                    writer.writerow(row)
            except:
                with open('Project/data/movies_metrics_fails.txt', 'a') as f:
                    f.write(f'{id}\n')
        print(f'Time taken: {time.time()-t0}')
    else:
        with open('Project/data/movies_metrics_fails.txt', 'r') as f:
            ids = f.read().splitlines()
        print(ids)
        for id in ids:
            row = get_data(id)
            with open('Project/data/movies_metrics.csv','a', newline='') as fd:
                writer = csv.writer(fd, delimiter=chr(255))
                writer.writerow(row)


if __name__ == '__main__':
    main(True)