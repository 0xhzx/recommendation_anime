import pandas as pd
import numpy as np


class naive_approach:
    def __init__(self, anime):
        self.anime = anime

    def get_unique_genres(self):
        # find out how many unique genres there are
        genres = self.anime['Genres'].str.split(', ')
        unique_genres = set()
        for genre_list in genres:
            for genre in genre_list:
                unique_genres.add(genre)
        unique_genres = list(unique_genres)
        unique_genres.sort()
        return unique_genres
    
    def get_top_k(self, genre_list, k=10):
        anime = self.anime
        
        unique_anime_list = set()
        for genre in genre_list:
            for i in range(len(self.anime)):
                if genre in anime['Genres'][i]:
                    unique_anime_list.add(anime['Name'][i])
                    
        unique_anime_list = list(unique_anime_list)
        # filter out the anime that in the unique_anime_list
        anime = anime[anime['Name'].isin(unique_anime_list)]
        # drop score = unknown
        anime = anime[anime['Score'] != 'Unknown']
        anime = anime.sort_values(by='Score', ascending=False)
        return anime.head(k)


if __name__ == '__main__':
    # Load data
    anime = pd.read_csv('../data/anime.csv')
    anime = anime.dropna()
    # Create an instance of naive_approach
    naive = naive_approach(anime)
    # Get unique genres
    unique_genres = naive.get_unique_genres()
    # Get top 10 anime for each genre
    candidate_genres = ['Action', 'Adventure']
    top_k = naive.get_top_k(candidate_genres, 10)
    print(top_k)
    