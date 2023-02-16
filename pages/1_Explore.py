import ast
import numpy as np
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="Explore", page_icon=":black_nib:", layout="wide")

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

st.sidebar.header("Explore")
st.sidebar.info("Exploration Page lets you explore the various datasets provided by MovieLens and TMDB API.")

st.header('Exploration')

col1, col2 = st.columns(2)
with col1:
    data = st.selectbox("Select The Dataset You'd Like To Explore:",
                    ('Movies', 'Desc_movies', 'Desc2_movies', 'Genres', 'Posters', 'Trailers', 'Links', 'Ratings'))
    generate = st.button('Generate Profile')
with col2:
    if data:
        st.info(data + ':')
        st.info('~ ' + ' • '.join(list(eval(data.lower()).columns)) + ' ~')
if data == 'Ratings':
  st.error('Caution: Large File')
st.dataframe(eval(data.lower()).head(1000), 100, 250, use_container_width=True)
if generate:
  if data != 'Desc_movies' or data != 'Desc2_movies' or data != 'Posters' or data != 'Trailers':
    profile = ProfileReport(eval(data.lower()),
                              explorative = True,
                              dark_mode = True,
                              lazy = False,
                              title = data,
                              dataset = {
                              "description": "MovieLens 25M movie ratings. Stable benchmark dataset. 25 million ratings and one million tag applications applied to 62,000 movies by 162,000 users. Includes tag genome data with 15 million relevance scores across 1,129 tags.",
                              "provider": "GroupLens",
                              "release_year": "2019",
                              "url": "https://grouplens.org/datasets/movielens/25m/",
                              })
  else:
    profile = ProfileReport(eval(data.lower()),
                              explorative = True,
                              dark_mode = True,
                              lazy = False,
                              title = data,
                              dataset = {
                              "description": "Scraped using the TMDB API.",
                              "provider": "TMDB",
                              "url": "https://www.themoviedb.org/documentation/api",
                              })
  st_profile_report(profile)