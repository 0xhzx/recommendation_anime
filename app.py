import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
import torch
from script.ncf import NNHybridFiltering, generate_recommendations, load_data, split_data

@st.cache_data()
def load_env_data():
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


def prepare_model():
    ratings, onehot_encode_map = load_data()
    X, y, X_train, X_val, X_test, y_train, y_val, y_test = split_data(ratings)
    n_users = X.loc[:,'user_id'].max()+1
    n_items = X.loc[:,'anime_id'].max()+1
    n_genres = 47

    model = NNHybridFiltering(n_users,
                        n_items,
                        n_genres,
                        embdim_users=50,
                        embdim_items=50,
                        embdim_genres=25,
                        n_activations = 100,
                        rating_range=[1.,10.])
    model.load_state_dict(torch.load('./model/ncf_model.pt', map_location=torch.device('cpu')))
    anime_test = pd.read_csv("./data/anime_test.csv")
    X_test = pd.read_csv("./data/X_test.csv")

    
    return model, anime_test, X_test

def recommend_ncf(selected_anime, selected_userid):
    ncf_model, anime_test, X_test = prepare_model()
    device = 'cpu'
    # selected_anime = anime_test[anime_test['Name'].isin(selected_anime)] # df with selected anime
    history_df = anime_test[anime_test['user_id'] == selected_userid]
    recs = generate_recommendations(anime_test,X_test,ncf_model,selected_userid,device) # names of recommended anime
    return recs, history_df
    


def create_genre_tags(genre_string):
    genres = genre_string.split(', ')
    colored_genres = [f'<span style="color: #{hash(genre) % 0xFFFFFF:06x};">{genre}</span>' for genre in genres]
    return ' '.join(colored_genres)


def main():
    print("Start................")
    anime_mapping, similarity, titles = load_env_data()
    # anime_mapping, similarity, titles = {},[],{}
    X_test = pd.read_csv('./data/X_test.csv')
    anime_test = pd.read_csv('./data/anime_test.csv')
    merged_df = X_test.merge(anime_test, on='user_id')
    common_user_ids = merged_df['user_id'].unique()[:10]

    st.title('Anime Recommendation System')
    selected_userid = st.selectbox('Type a user', options=common_user_ids)
    selected_anime = st.multiselect('Type a Anime', options=titles)
    if st.button('Recommend'):
        # recommended_anime_names, recommended_anime_genres = recommender(selected_anime, anime_mapping, similarity)
        recommended_anime_names, history_df = recommend_ncf(selected_anime, selected_userid)

        # Display the recommended anime
        # size = len(recommended_anime_names)
        # columns = st.columns(size)

        # for i, col in enumerate(columns):
        #     with col:
        #         st.markdown(f"**{recommended_anime_names[i]}**")
        #         # use Markdown show color label
        #         genre_tags_html = create_genre_tags(anime_mapping[anime_mapping['Name'] == recommended_anime_names[i]]['Genres'].values[0])
        #         st.markdown(genre_tags_html, unsafe_allow_html=True)
        selected_columns = ['Name', 'Score', 'Genres', 'rating']
        result_df = history_df[selected_columns]
        st.markdown(f"**{selected_userid} user observed history**")
        print("history_df:",history_df.shape)
        # print("history_df:",history_df)
        st.write(result_df)

        size = len(recommended_anime_names)
        columns = st.columns(5)  

        for i in range(size):
            with columns[i % 5]:  
                st.markdown(f"**{recommended_anime_names[i]}**")
                genre_tags_html = create_genre_tags(anime_mapping[anime_mapping['Name'] == recommended_anime_names[i]]['Genres'].values[0])
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