import re
import ast
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pandas.io.parsers.python_parser import count_empty_vals

st.set_page_config(page_title="Visualize", page_icon=":mag:", layout="wide")

@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  return df

@st.cache_data
def load_scraped_data(filename):
  df = pd.read_csv(filename, lineterminator='\n')
  return df

# Might also import the data from '1_Explore.py' using importlib

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
 
st.sidebar.header("Visualization")
st.sidebar.info("Visualization Page lets you visualize various aspects and traits within the datasets.")

st.header('Visualization')

tab1, tab2, tab3, tab4 = st.tabs(["Popularity", "Seggregation", "Distribution", "Co-appearance Network"])
with tab1:

    # Static plots are cached to save time

    @st.cache_data
    def mostPopularMovies():

        # Top 10 most popular movies

        mostPopular = desc_movies.iloc[:, :3].sort_values('popularity', ascending=False)
        fig = px.bar(mostPopular[:50], y='title', x='popularity', color='popularity', title='Top 50 Most Popular Movies', color_continuous_scale=px.colors.sequential.thermal)
        return fig

    @st.cache_data
    def mostVotedMovies():

        # Top 10 most voted movies

        db = pd.merge(desc_movies, desc2_movies, on='movieId', how='inner')
        mostVoted = db.loc[:, db.columns.intersection(['movieId', 'title', 'vote_count'])].sort_values('vote_count', ascending=False)
        fig = px.bar(mostVoted[:50], y='title', x='vote_count', color='vote_count', title='Top 50 Most Voted Movies', color_continuous_scale=px.colors.sequential.deep)
        return fig

    
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.plotly_chart(mostPopularMovies(), use_container_width=True)
    with col2:
        st.plotly_chart(mostVotedMovies(), use_container_width=True)

with tab2:

    # Static plots are cached to save time

    @st.cache_data
    def genSeg():

        # Genre seggregation of movies

        genLabel = list(genres.iloc[:, 2:].columns)
        genCount = []
        for i in range(len(genLabel)):
            genCount.append(np.sum(genres.iloc[:, i+2]))
        db = pd.DataFrame({'genre': genLabel, 'value': genCount})
        fig = px.bar(db, x='genre', y='value', color='value', title='Genre Seggregation of Movies', color_continuous_scale=px.colors.sequential.RdBu)
        return fig

    @st.cache_data
    def keySeg():

        # Keyword seggregation of movies
        
        db = pd.merge(desc_movies, desc2_movies, on='movieId', how='inner')
        db = db.loc[:, db.columns.intersection(['movieId', 'title', 'keywords'])]
        keywords = {}
        for i in db['keywords'].values:
            for j in i.strip("']['").split("', '"):
                if j in keywords:
                    keywords[j] += 1
                else:
                    keywords[j] = 1
        keywords = dict(sorted(keywords.items(), key=lambda item:item[1], reverse=True))
        keywords = dict(sorted(keywords.items(), key=lambda item:item[1], reverse=True)[1:101])     # First value is an empty string
        keys = list(keywords.keys())
        counts = list(keywords.values())
        fig = px.bar(x=keys, y=counts, color=counts, title='Keyword Seggregation of Movies', color_continuous_scale=px.colors.sequential.Teal)
        return fig

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.plotly_chart(genSeg(), use_container_width=True)
    with col2:
        st.plotly_chart(keySeg(), use_container_width=True)

with tab3:

    # Static plots are cached to save time

    @st.cache_data
    def avgVoteDist():
        
        # Average vote distribution

        db = pd.merge(desc_movies, desc2_movies, on='movieId', how='inner')
        db = db.loc[:, db.columns.intersection(['movieId', 'title', 'vote_average'])]
        voteDist = db.groupby('vote_average')['vote_average'].count().reset_index(name='vote_dist').sort_values('vote_dist', ascending=False)
        fig = px.scatter(voteDist, x='vote_average', y='vote_dist', size='vote_dist', color='vote_dist', title='Average Vote Distribution', color_continuous_scale=px.colors.sequential.Burg, marginal_x='histogram', marginal_y='rug')
        return fig
    
    @st.cache_data
    def popCountDist():
        db = pd.merge(desc_movies, desc2_movies, on='movieId', how='inner')
        popCountDist = db.loc[:1000, db.columns.intersection(['movieId', 'title', 'popularity', 'vote_count'])]
        fig = px.scatter(popCountDist, x='popularity', y='vote_count', size='vote_count', color='vote_count', title='Cross Appearance of Popularity and Vote Count', color_continuous_scale=px.colors.sequential.Darkmint, marginal_x='rug', marginal_y='rug')
        return fig

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.plotly_chart(avgVoteDist(), use_container_width=True)
    with col2:
        st.plotly_chart(popCountDist(), use_container_width=True)

with tab4:

    # Coappearance of actors
    def casts(x):
        return ', '.join(ast.literal_eval(x.casts))

    def network(graphWeights, actors):

        graphWeights = graphWeights[graphWeights['source'] != graphWeights['target']]
        group, count = 1, 1

        nodes = []
        for key, val in actors.items():
            node = {}
            node['name'] = key
            node['group'] = group
            if count == 2:
                count = 1
                group += 1
            nodes.append(node)
            count += 1

        links = []
        for i in graphWeights.values:
            link = {}
            link['source'] = actors[i[0]]
            link['target'] = actors[i[1]]
            link['value'] = i[2]
            links.append(link)
        N=len(nodes)
        L=len(links)

        Edges=[(links[k]['source'], links[k]['target']) for k in range(L)]
        G=ig.Graph(Edges, directed=False)
        labels=[]
        group=[]
        for node in nodes:
            labels.append(node['name'])
            group.append(node['group'])
        layt=G.layout('kk', dim=3)

        Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
        Yn=[layt[k][1] for k in range(N)]# y-coordinates
        Zn=[layt[k][2] for k in range(N)]# z-coordinates
        Xe=[]
        Ye=[]
        Ze=[]
        for e in Edges:
            Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
            Ye+=[layt[e[0]][1],layt[e[1]][1], None]
            Ze+=[layt[e[0]][2],layt[e[1]][2], None]
        Gnx = nx.Graph(Edges) 
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(Gnx.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append('# of connections: '+str(len(adjacencies[1])))
        trace1=go.Scatter3d(x=Xe,
                    y=Ye,
                    z=Ze,
                    mode='lines',
                    line=dict(color='rgb(255,255,255)', width=0.5),
                    hoverinfo='text'
                    )
        trace2=go.Scatter3d(x=Xn,
                    y=Yn,
                    z=Zn,
                    mode='markers',
                    name='actors',
                    marker=dict(symbol='circle',
                                    size=10,
                                    color=node_adjacencies,
                                    reversescale=True,
                                    colorscale='Darkmint',
                                    colorbar=dict(title='Number of Connections', title_side='right'),
                                    line=dict(color='rgb(50,50,50)', width=0.5)
                                    ),
                    text=labels,
                    hoverinfo='text'
                    )
        axis=dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
        layout = go.Layout(
                title="Network of Coappearances of Casts in the Movies",
                showlegend=False,
                scene=dict(
                    xaxis=dict(axis),
                    yaxis=dict(axis),
                    zaxis=dict(axis),
                ),
                margin=dict(
                    t=100
                ),
                hovermode='closest',
                annotations=[
                    dict(
                    showarrow=False,
                        text="",
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=0.1,
                        xanchor='left',
                        yanchor='bottom',
                        font=dict(
                        size=14
                        )
                        )
                    ])
        data=[trace1, trace2]
        return data, layout

    tempdb = pd.concat([desc_movies.iloc[:, :-3], desc_movies.apply(casts, axis=1)], axis=1)
    tempdb = tempdb.rename(columns={0: 'casts'})
    tempdb = tempdb[tempdb['casts'] != '']
    tempdb = tempdb[tempdb['casts'].str.contains(u', ')]

    actors = {}
    casts = list(tempdb.casts)
    for i in casts:
        j = i.split(', ')
        if j[0] not in actors:
            actors[j[0]] = j[1:]
        if j[0] in actors:
            actors[j[0]] += j[1:]

    graphActors = {key: val for key, val in actors.items() if val != []}
    count = 0
    actors = {}
    for i in casts:
        for j in i.split(', '):
            if j not in actors:
                actors[j] = count
                count += 1
    source = []
    target = []
    weight = []
    for key, value in graphActors.items():
        worked = {}
        source += [key] * len(set(value))
        for i in value:
            if i not in worked:
                worked[i] = 1
            else:
                worked[i] += 1
        target += worked.keys()
        weight += worked.values()
    graphWeights = pd.DataFrame({'source': source, 'target': target, 'weight': weight})

    choices = list(actors.keys())
    choice = st.selectbox('Choose An Actor:', sorted(choices))
    tempdb = graphWeights[(graphWeights['source'] == choice) | (graphWeights['target'] == choice)]
    tempact = {}
    count = 0
    for i in tempdb.values:
        if i[0] not in tempact:
            tempact[i[0]] = count
            count += 1
        if i[1] not in tempact:
            tempact[i[1]] = count
            count += 1
    data, layout = network(tempdb, tempact)
    fig=go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)