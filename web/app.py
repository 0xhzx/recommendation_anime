import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

@st.cache_data()
def load_data():
    # global vars
    anime_mapping = pd.read_csv('./data/anime_mapping.csv',index_col='anime_id')
    similarity = pd.read_csv('./data/small_csmatrix.csv',index_col='anime_id')
    titles = anime_mapping['Name'][20:50]
    print("Loading ends")
    return anime_mapping, similarity, titles


def fetch_poster(anime_id):
    # search the local storage images to show
    full_path = ''
    
    return full_path


def recommender(anime, anime_mapping ,similarity):
    anime_index = anime_mapping[anime_mapping['Name'] == anime].index[0]
    print("anime index is:", anime_index)
    distance = similarity.loc[anime_index]
    print("Start calculating")
    anime_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:8]
    print(anime_list)
    print("Search ends")
    anime_recommend = []
    anime_recommend_posters = []
    for i in anime_list:
        anime_id = anime_mapping.loc[i[0]]['anime_id']
        if anime_id in anime_mapping.index:
            anime_recommend.append(anime_mapping.loc[i[0]]['Name'])
            anime_recommend_posters.append(fetch_poster(anime_id))

    return anime_recommend, anime_recommend_posters


def main():
    print("Start................")
    anime_mapping, similarity, titles = load_data()
    # anime_mapping, similarity, titles = {},[],{}

    st.title('Anime Recommendation System')
    selected_anime = st.selectbox('Type a Anime', options=titles)
    if st.button('Recommend'):
        recommended_anime_names, recommended_anime_posters = recommender(selected_anime, anime_mapping, similarity)

        # Display the recommended anime
        size = len(recommended_anime_names)
        col1, col2, col3, col4, col5 = st.columns(size)
        idx = 0
        with col1:
            st.text(recommended_anime_names[idx])
            st.image(recommended_anime_posters[idx])
        with col2:
            idx += 1
            st.text(recommended_anime_names[idx])
            st.image(recommended_anime_posters[idx])
        with col3:
            idx += 1
            st.text(recommended_anime_names[idx])
            st.image(recommended_anime_posters[idx])
        with col4:
            idx += 1
            st.text(recommended_anime_names[idx])
            st.image(recommended_anime_posters[idx])
        with col5:
            idx += 1
            st.text(recommended_anime_names[idx])
            st.image(recommended_anime_posters[idx])

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