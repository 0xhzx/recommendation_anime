import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

@st.cache_data()
def load_data():
    # global vars
    anime_mapping = pd.read_csv('./data/anime_mapping.csv')
    similarity = pd.read_csv('./data/small_csmatrix.csv',index_col='anime_id')
    titles = anime_mapping['Name'][20:50]
    print("Loading ends")
    return anime_mapping, similarity, titles


def fetch_poster(anime_id):
    # search the local storage images to show
    full_path = ''
    
    return full_path


def recommender(anime, anime_mapping ,similarity):
    anime_index = anime_mapping[anime_mapping['Name'] == anime]['anime_id'].values[0]
    print("anime index is:", anime_index)
    distance = similarity.loc[anime_index]
    print("Start calculating")
    # anime_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:8]
    anime_list = distance.sort_values(ascending=False).index.values[1:6]
    print(anime_list)
    print("Search ends")
    anime_recommend = []
    # anime_recommend_posters = []
    anime_recommend_genres = []
    for anime_id in anime_list:
        anime_id = int(anime_id)
        # print("add anime id:", anime_id)
        anime_recommend.append(anime_mapping[anime_mapping['anime_id'] == anime_id]['Name'].values[0])
        anime_recommend_genres.append(anime_mapping[anime_mapping['anime_id']== anime_id]['Genres'].values[0])
        # anime_recommend_posters.append([])
        # anime_recommend_posters.append(fetch_poster(anime_id))

    return anime_recommend, anime_recommend_genres

def create_genre_tags(genre_string):
    genres = genre_string.split(', ')
    colored_genres = [f'<span style="color: #{hash(genre) % 0xFFFFFF:06x};">{genre}</span>' for genre in genres]
    return ' '.join(colored_genres)


def main():
    print("Start................")
    anime_mapping, similarity, titles = load_data()
    # anime_mapping, similarity, titles = {},[],{}

    st.title('Anime Recommendation System')
    selected_anime = st.selectbox('Type a Anime', options=titles)
    if st.button('Recommend'):
        recommended_anime_names, recommended_anime_genres = recommender(selected_anime, anime_mapping, similarity)

        # Display the recommended anime
        size = len(recommended_anime_names)
        columns = st.columns(size)

        for i, col in enumerate(columns):
            with col:
                st.markdown(f"**{recommended_anime_names[i]}**")
                # use Markdown show color label
                genre_tags_html = create_genre_tags(recommended_anime_genres[i])
                st.markdown(genre_tags_html, unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    hide_decoration_bar_style = '''
        <style>
            header {visibility: hidden;}
        </style>
    '''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
    main()