import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def genres_vectorized(df):
    vec = CountVectorizer()
    genres_vec = vec.fit_transform(df['Genres'])
    return genres_vec

def csmartrix(genres_vec, df):
    csmatrix = cosine_similarity(genres_vec)
    csmatrix = pd.DataFrame(csmatrix,columns=df.anime_id,index=df.anime_id)
    return csmatrix

def load_data(df):
    X = df.drop(labels=['rating','Genres'],axis=1)
    y = df['rating']
    X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0, test_size=0.2)
    return X_train, X_val, y_train, y_val

def predict_rating(user_item_pair,simtable=csmatrix,X_train=X_train, y_train=y_train):
    anime_to_rate = user_item_pair['anime_id']
    user = user_item_pair['user_id']
    # Filter similarity matrix to only movies already reviewed by user
    anime_watched = X_train.loc[X_train['user_id']==user, 'anime_id'].tolist()
    simtable_filtered = simtable.loc[anime_to_rate,anime_watched]
    # Get the most similar movie already watched to current movie to rate
    most_similar_watched = simtable_filtered.index[np.argmax(simtable_filtered)]
    # Get user's rating for most similar movie
    idx = X_train.loc[(X_train['user_id']==user) & (X_train['anime_id']==most_similar_watched)].index.values[0]
    most_similar_rating = y_train.loc[idx]
    return most_similar_rating

def RMSE(ratings_valset):
    val_rmse = np.sqrt(mean_squared_error(y_val,ratings_valset))
    return val_rmse

def generate_recommendations(user,simtable,ratings):
    # Get top rated movie by user
    user_ratings = ratings.loc[ratings['user_id']==user]
    user_ratings = user_ratings.sort_values(by='rating',axis=0,ascending=False)
    topratedanime = user_ratings.iloc[0,:]['anime_id']
    topratedanime_title = df2.loc[df2['anime_id']==topratedanime,'Name'].values[0]
    # Find most similar movies to the user's top rated movie
    sims = simtable.loc[topratedanime,:]
    mostsimilar = sims.sort_values(ascending=False).index.values
    # Get 10 most similar movies excluding the movie itself
    mostsimilar = mostsimilar[1:11]
    # Get titles of movies from ids
    mostsimanime_names = []
    for anime in mostsimilar:
        mostsimanime_names.append(df2.loc[df2['anime_id']==anime,'Name'].values[0])
    return topratedanime_title, mostsimanime_names

if __name__ == '__main__':
    # data processing
    df = pd.read_csv('../data/animelist_reduced.csv')
    df2 = pd.read_csv('../data/anime_with_synopsis.csv')
    df2.rename(columns={'MAL_ID': 'anime_id'}, inplace=True)
    df2 = df2[['anime_id', 'Name', 'Genres']]
    grouped_df = df.groupby('user_id').agg(
    anime_watched_amount=pd.NamedAgg(column='anime_id', aggfunc='count')
    )
    # remove users who have watched less than 10 anime
    less_than_10 = grouped_df[grouped_df['anime_watched_amount'] <= 10].index
    df_filtered = df[~df['user_id'].isin(less_than_10)]
    grouped_df = df.groupby('anime_id').agg(
    anime_watched_amount=pd.NamedAgg(column='user_id', aggfunc='count')
    )

    merged_df = pd.merge(df_filtered, df2[['anime_id', 'Genres']], on='anime_id', how='left')
    ratings = merged_df[['user_id', 'anime_id', 'Genres', 'rating']]
    ratings['Genres'] = ratings['Genres'].fillna('Hentai')
    df_unique_anime_id = ratings.drop_duplicates(subset='anime_id', keep='first')
    df_unique_anime_id = df_unique_anime_id[['anime_id', 'Genres']]


    genres_vec = genres_vectorized(df_unique_anime_id)
    csmartrix = csmartrix(genres_vec, df_unique_anime_id)
    X_train, X_val, y_train, y_val =load_data(ratings)
    ratings_valset = X_val.apply(lambda x: predict_rating(x),axis=1)
    RMSE = RMSE(ratings_valset)
    user = 34
    topratedmovie, recs = generate_recommendations(user,simtable=csmatrix,ratings=ratings)
    print("User's highest rated movie was {}".format(topratedmovie))
    for i,rec in enumerate(recs):
        print('Recommendation {}: {}'.format(i,rec))