import re
import ast
import joblib
import random
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from surprise import dump
from tmdbv3api import TMDb
from tmdbv3api import Movie
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Recommend", page_icon=":movie_camera:", layout="wide")

@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  return df

@st.cache_data
def load_scraped_data(filename):
  df = pd.read_csv(filename, lineterminator='\n')
  return df

links = load_data('./data/links.csv')
movies = load_data('./data/movies.csv')
genres = load_data('./data/genres.csv')
ratings = load_data('./data/ratings.csv')
try:
  posters = load_data('./data/posters.csv')
  trailers = load_data('./data/trailers.csv')
  desc_movies = load_data('./data/desc_movies.csv')
  desc2_movies = load_data('./data/desc2_movies.csv')
except:
  posters = load_scraped_data('./data/posters.csv')
  trailers = load_scraped_data('./data/trailers.csv')
  desc_movies = load_scraped_data('./data/desc_movies.csv')
  desc2_movies = load_scraped_data('./data/desc2_movies.csv')
 
st.sidebar.header("Recommendation")
st.sidebar.info("Recommendation Page lets you ask for recommendations on given inputs, along with a few other popular movies.")

st.header('Recommendation')

tab1, tab2 = st.tabs(['Popularity Based', 'Content/Collabortive Based'])
def mov(res, ind):

    # Defines the container format

    st.image(res[ind]['Poster'])
    st.subheader(res[ind]['Title'])
    st.caption(res[ind]['Overview'])
    st.write(' • '.join(ast.literal_eval(res[ind]['Cast'])))

def container(desc):

    # Defines the container layout
    col1, col2, col3, col4, col5 = st.columns(5)
    try:
        with col1:
            mov(desc, 0)
        with col2:
            mov(desc, 1)
        with col3:
            mov(desc, 2)
        with col4:
            mov(desc, 3)
        with col5:
            mov(desc, 4)
    except:
        pass

with tab1:

    def popularMeasureTMDB(db, num):

        # Selects top 500 most popular movies and returns a random sample of 5 from them

        suggestions = []
        popularity = db.sort_values('popularity', ascending=False)
        res = pd.merge(popularity[:500], posters, on='movieId', how='inner')
        res = res.sample(n = num)[['movieId', 'title', 'overview', 'posters', 'casts']]
        for i in res.values:
            result = {}
            result['Title'] = i[1]
            result['Overview'] = i[2]
            result['Poster'] = i[3]
            result['Cast'] = i[4]
            suggestions.append(result)
        return suggestions

    desc = popularMeasureTMDB(desc_movies, 5)
    container(desc)

    # Genre wise popular selection

    st.subheader('Select A Genre:')
    gen = st.select_slider('Genre', options=list(genres.iloc[:, 2:-1]), label_visibility='collapsed')
    st.subheader(gen)
    tempdb = pd.merge(genres[genres[gen] == 1], links, on='movieId', how='inner')
    tempdb = tempdb[['tmdbId']].rename(columns={'tmdbId': 'movieId'})
    tempdb = pd.merge(tempdb, desc_movies[['movieId', 'title', 'overview', 'popularity', 'casts']], on='movieId', how='inner')

    desc = popularMeasureTMDB(tempdb, 5)
    container(desc)

with tab2:

    # Contents is the sampled dataframe on which the vectorizer was fitted
    try:
        tempdb = load_data('./data/contents.csv')

    except:
        tempdb = load_scraped_data('./data/contents.csv')
    count = joblib.load('./systems/count.pkl')

    # The following dataset is required for collaborative recommendations
    temp = tempdb
    temp['title'] = temp['title'].apply(lambda x: x[:-7])

    def overTitle(title):

        # Retrieves only the title and overview of a movie

        tmdb = TMDb()
        movie = Movie()
        tmdb.api_key = API_KEY
        tmdb.language = 'en'
        tmdb.debug = True

        mov = movie.search(title)[0]
        title = mov.title.lower()
        overview = mov.overview.lower()
        result = re.sub("[^a-zA-Z0-9 ]", "", title + ' ' + overview)
        return mov.title, result

    def descTitle(title):

        # Retrieves title, overview, genres, casts and keywords of a movie

        tmdb = TMDb()
        movie = Movie()
        tmdb.api_key = API_KEY
        tmdb.language = 'en'
        tmdb.debug = True
        mov = movie.details(movie.search(title)[0].id)
        title = mov.title.lower()
        overview = mov.overview.lower()
        genres = ' '.join(i.name.lower() for i in mov.genres)
        casts = ' '.join(i.name.replace(' ', '').lower() for i in mov.casts.cast[:5])
        keywords = ' '.join(i.name.replace(' ', '').lower() for i in mov.keywords.keywords)
        result = re.sub("[^a-zA-Z0-9 ]", "", title + ' ' + overview + ' ' +  casts + ' ' +  genres + ' ' +  keywords)
        return mov.title, result

    def contextBasedRecommendations(title, num):

        # Fetches top 10 results based on similarity (not more as it may lose relevance) but returns 5 random ones

        try:
            try:
                title, desc = descTitle(title)
            except:
                title, desc = overTitle(title)
        except:
            # In case the API fails to retrieve results, displaying some popular movies
            # While returning the title of the most popular movie for collaborative recommendation
            return 'Avatar (2009)', popularMeasureTMDB(desc_movies, num)

        query_vec = count.transform([desc]) 
        count_matrix = sparse.load_npz('./systems/count_matrix.npz')                  # Transforming the modified title into a query vec using the fitted count vectorizer
        similarity = cosine_similarity(query_vec, count_matrix).flatten()             # Computing cosine similarity measure
        inx = np.argsort(similarity)[::-1][0:num]                                     # Getting the most relevant recommendations
        res = tempdb.iloc[inx][['movieId', 'title', 'overview', 'casts']]
        res = pd.merge(res, posters, on='movieId', how='inner')[['movieId', 'title', 'overview', 'posters', 'casts']]

        suggestions = []
        for i in res.values:
            result = {}
            result['Title'] = i[1]
            result['Overview'] = i[2]
            result['Poster'] = i[3]
            result['Cast'] = i[4]
            suggestions.append(result)
        return title, suggestions

    def collaborativeBasedRecommendations(title, num):

        # Randomly selects users who have rated the provided title and then finds their highest rated movies before randomly showing it

        try:
            id = float(list(t[t['title'] == ''].sort_values('movieId')['movieId'])[0])
            id = int(links[links["tmdbId"] == id]['movieId'])
            userList = list(ratings[ratings['movieId'] == id].sort_values('rating', ascending=False)['userId'])
        except:
            mostUsers = ratings.iloc[:, :1].groupby('userId')['userId'].count().reset_index(name='count')
            mostUsers = mostUsers.sort_values('count')
            userList = list(mostUsers['userId'])
        
        _, svd = dump.load('./systems/svd.pkl')

        movieIdx = np.unique(tempdb.movieId)
    
        predictions = []
        suggestions = []
        for userId in (random.sample(userList, 10) if len(userList)>10 else userList):
            for i in movieIdx:
                predictions.append(svd.predict(userId, i))

            # First map the predictions for the given user
            top_n = defaultdict(list)
            for uid, iid, true_r, est, _ in predictions:
                top_n[uid].append((iid, est))

            # Then sort the predictions for the given user and retrieve the k highest ones.
            for uid, user_ratings in top_n.items():
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n[uid] = user_ratings[:num]

            uid = list(top_n.items())[0][0]
            user_ratings = list(top_n.items())[0][1]
            movieIds = [iid for (iid, _) in user_ratings]
            ids = pd.DataFrame({'movieId': movieIds})
            res = pd.merge(movies, ids, on='movieId', how='inner')
            res = pd.merge(links, res, on='movieId', how='inner')[['tmdbId', 'genres']].rename(columns={'tmdbId': 'movieId'})
            res = pd.merge(desc_movies, res, on='movieId', how='inner')
            res = pd.merge(res, posters, on='movieId', how='inner')[['movieId', 'title', 'overview', 'posters', 'casts']]

            for i in res.values:
                result = {}
                result['Title'] = i[1]
                result['Overview'] = i[2]
                result['Poster'] = i[3]
                result['Cast'] = i[4]
                suggestions.append(result)
        rec = []
        for i in suggestions:
            if i not in rec:
                rec.append(i)
        return (random.sample(rec, num) if len(rec)>num else rec)


    st.error("Caution: Click on 'Generate Recommendations' once the details have been entered")
    form = st.form(key = 'rec')
    name = form.text_input('Enter Your Movie Title:', 'The Dark Knight')
    # API_KEY = form.text_input('Enter Your TMDb API Key:')
    generate = form.form_submit_button('Generate Recommendations:')

    if generate:
        title, desc = contextBasedRecommendations(name, 5)
        container(desc)

        st.subheader('Users Also Liked')
    
        desc = collaborativeBasedRecommendations(title, 5)
        container(desc)