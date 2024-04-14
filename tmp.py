import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Cache data to optimize load times
@st.cache
def load_data():
    anime_mapping = pd.read_csv('./data/anime_mapping.csv')
    similarity = pd.read_csv('./data/small_csmatrix.csv', index_col='anime_id')
    titles = anime_mapping['Name'][20:50]
    return anime_mapping, similarity, titles

def recommender(anime, anime_mapping, similarity):
    anime_index = anime_mapping[anime_mapping['Name'] == anime]['anime_id'].values[0]
    distance = similarity.loc[anime_index]
    anime_list = distance.sort_values(ascending=False).index.values[1:6]
    anime_recommend = [anime_mapping[anime_mapping['anime_id'] == anime_id]['Name'].values[0] for anime_id in anime_list]
    anime_recommend_genres = [anime_mapping[anime_mapping['anime_id'] == anime_id]['Genres'].values[0] for anime_id in anime_list]
    return anime_recommend, anime_recommend_genres

def create_genre_tags(genre_string):
    genres = genre_string.split(', ')
    colored_genres = [f'<span style="color: #{hash(genre) % 0xFFFFFF:06x};">{genre}</span>' for genre in genres]
    return ' '.join(colored_genres)

def main():
    anime_mapping, similarity, titles = load_data()
    st.title('Anime Recommendation System')
    selected_anime = st.selectbox('Type an Anime', options=titles)

    if st.button('Recommend'):
        recommended_anime_names, recommended_anime_genres = recommender(selected_anime, anime_mapping, similarity)
        columns = st.columns(len(recommended_anime_names))

        for i, col in enumerate(columns):
            with col:
                st.markdown(f"**{recommended_anime_names[i]}**")
                genre_tags_html = create_genre_tags(recommended_anime_genres[i])
                st.markdown(genre_tags_html, unsafe_allow_html=True)

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
            #MainMenu, footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    main()
