import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

# global vars
anime_mapping = pd.read_csv('./data/anime_mapping.csv',index_col='anime_id')
similarity = pd.read_csv('./data/csmatrix.csv',index_col='anime_id')
titles = anime_mapping['Name'][0:20]
print("Loading ends")


def fetch_poster(anime_id):
    # search the local storage images to show
    full_path = ''
    
    return full_path


def recommender(anime):
    anime_index = anime_mapping[anime_mapping['Name'] == anime].index[0]
    distance = similarity[anime_index]
    print("Start calculating")
    anime_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:11]
    print(anime_list)
    anime_recommend = []
    anime_recommend_posters = []
    for i in anime_list:
        anime_id = anime_mapping.iloc[i[0]]['anime_id']
        anime_recommend.append(anime_mapping.iloc[i[0]]['Name'])
        anime_recommend_posters.append(fetch_poster(anime_id))

    return anime_recommend, anime_recommend_posters


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


st.title('Anime Recommendation System')
selected_anime = st.selectbox('Type a Anime', options=titles)
if st.button('Recommend'):
    recommended_anime_names, recommended_anime_posters = recommender(selected_anime)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_anime_names[0])
        st.image(recommended_anime_posters[0])
    with col2:
        st.text(recommended_anime_names[1])
        st.image(recommended_anime_posters[1])
    with col3:
        st.text(recommended_anime_names[2])
        st.image(recommended_anime_posters[2])
    with col4:
        st.text(recommended_anime_names[3])
        st.image(recommended_anime_posters[3])
    with col5:
        st.text(recommended_anime_names[4])
        st.image(recommended_anime_posters[4])

    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        st.text(recommended_anime_names[5])
        st.image(recommended_anime_posters[5])
    with col7:
        st.text(recommended_anime_names[6])
        st.image(recommended_anime_posters[6])
    with col8:
        st.text(recommended_anime_names[7])
        st.image(recommended_anime_posters[7])
    with col9:
        st.text(recommended_anime_names[8])
        st.image(recommended_anime_posters[8])
    with col10:
        st.text(recommended_anime_names[9])
        st.image(recommended_anime_posters[9])