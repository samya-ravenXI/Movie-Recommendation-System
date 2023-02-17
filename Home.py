import base64
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Home",
    page_icon=":earth_africa:",
)

def load_bootstrap():
    return st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

st.sidebar.markdown(
    '''
        <div style="border: thin solid black; border-radius: 5px;">
            <div style="background-image: url(data:image/png;base64,{}); background-repeat: no-repeat; height: 125px;">
                <a href='https://github.com/samya-ravenXI' style="position: absolute; z-index: 2; top: 50px; left: 20px;">
                    <img src="https://skillicons.dev/icons?i=github" alt="GitHub"/>
                </a>
                <h1 style="color: rgb(224, 224, 224); position: absolute; z-index: 2; left: 85px; font-family: sans-serif;">Movie Recommendation System</h1>
            </div>
            <div style="margin-top: 40px">
                <a href="https://github.com/samya-ravenXI/Movie-Recommendation-System" style="position: absolute; z-index: 2; top: 131px; left: 35px">
                    <img src="https://img.shields.io/badge/github-repo-white" alt="repo"/>
                </a>
                <a href="https://colab.research.google.com/drive/1J6hg-FvonxtgzQ71MNr11md5gFtiqmZV?usp=share_link" style="position: absolute; z-index: 2; top: 131px; right: 35px">
                    <img src="https://img.shields.io/badge/colab-notebook-orange" alt="repo"/>
                </a>
            </div>
        </div>
    '''.format(img_to_bytes('./icons/cover.jpg')),
    unsafe_allow_html=True)

with st.container():
    st.title("Movie Recommendation System")
    for i in range(2):
        st.markdown('#')
    st.caption('This projects uses MovieLens 25M Dataset along with TMDb API to explore and visualize trends in movies.')
    st.caption('It also contains a content based & collaborative based recommendation model using Vectorizer and SVD respectively.')
    st.caption('Tip: To use the content based/collaborative based recommendation system, a TMDb API Key is required.')

    for i in range(2):
        st.markdown('#')
    st.markdown('#####')
    st.markdown('---')

    col1, col2, col3 = st.columns(3, gap='large')
    with col1:
        st.empty()
        st.empty()
        st.markdown("<a href='https://docs.streamlit.io/library/get-started'><img src='data:image/png;base64,{}' class='img-fluid' width=100%/></a>".format(img_to_bytes('./icons/streamlit.png')), unsafe_allow_html=True)
    with col2:
        st.markdown("<a href='https://developers.themoviedb.org/3/getting-started/introduction'><img src='data:image/png;base64,{}' class='img-fluid' width=60%/></a>".format(img_to_bytes('./icons/tmdb.png')), unsafe_allow_html=True)
    with col3:
        st.markdown("<a href='https://colab.research.google.com/drive/1J6hg-FvonxtgzQ71MNr11md5gFtiqmZV?usp=share_link'><img src='data:image/png;base64,{}' class='img-fluid' width=50%/></a>".format(img_to_bytes('./icons/colab.png')), unsafe_allow_html=True)