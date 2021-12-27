"""
    @Author: Minh Duc
    @Since: 12/27/2021 3:44 PM
"""

import urllib3
import requests


def get_title_id(keyword):
    url = f"https://imdb-internet-movie-database-unofficial.p.rapidapi.com/search/{keyword}"
    url = urllib3.util.parse_url(url)
    headers = {
        'x-rapidapi-host': "imdb-internet-movie-database-unofficial.p.rapidapi.com",
        'x-rapidapi-key': "12ac25c0fbmshbac001bf55b1db2p15d186jsn334d832d1944"
    }

    response = requests.request("GET", url, headers=headers)
    data = response.json()

    # get the title id
    try:
        id = response.json()["titles"][0]["id"]
        return id
    except:
        return None


def get_film_info(title_id):
    url = "https://movie-database-imdb-alternative.p.rapidapi.com/"

    querystring = {"r": "json", "i": title_id}

    headers = {
        'x-rapidapi-host': "movie-database-imdb-alternative.p.rapidapi.com",
        'x-rapidapi-key': "12ac25c0fbmshbac001bf55b1db2p15d186jsn334d832d1944"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.json()
    info = {
        "Title": data["Title"], "Poster": data["Poster"], "Director": data["Director"],
        "Writer:": data["Writer"], "Actors": data["Actors"], "Plot": data["Plot"]
    }
    return info


