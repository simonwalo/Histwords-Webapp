import streamlit as st
import pickle
import gensim
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from adjustText import adjust_text


st.title('Historical Word Embeddings')

@st.cache(allow_output_mutation = True)
def load_data():
    models_all = {
        1800: pickle.load(open("./data/embeddings1800.pickle", 'rb')),
        1810: pickle.load(open("./data/embeddings1810.pickle", 'rb')),
        1820: pickle.load(open("./data/embeddings1820.pickle", 'rb')),
        1830: pickle.load(open("./data/embeddings1830.pickle", 'rb')),
        1840: pickle.load(open("./data/embeddings1840.pickle", 'rb')),
        1850: pickle.load(open("./data/embeddings1850.pickle", 'rb')),
        1860: pickle.load(open("./data/embeddings1860.pickle", 'rb')),
        1870: pickle.load(open("./data/embeddings1870.pickle", 'rb')),
        1880: pickle.load(open("./data/embeddings1880.pickle", 'rb')),
        1890: pickle.load(open("./data/embeddings1890.pickle", 'rb')),
        1900: pickle.load(open("./data/embeddings1900.pickle", 'rb')),
        1910: pickle.load(open("./data/embeddings1910.pickle", 'rb')),
        1920: pickle.load(open("./data/embeddings1920.pickle", 'rb')),
        1930: pickle.load(open("./data/embeddings1930.pickle", 'rb')),
        1940: pickle.load(open("./data/embeddings1940.pickle", 'rb')),
        1950: pickle.load(open("./data/embeddings1950.pickle", 'rb')),
        1960: pickle.load(open("./data/embeddings1960.pickle", 'rb')),
        1970: pickle.load(open("./data/embeddings1970.pickle", 'rb')),
        1980: pickle.load(open("./data/embeddings1980.pickle", 'rb')),
        1990: pickle.load(open("./data/embeddings1990.pickle", 'rb')),
    }
    return models_all

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load data.
models_all = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Data loaded to cache!")


keyword = st.text_input("Input term", "gay")


st.subheader('Most similar words per decade')

def simterms(keyword):

    data = pd.DataFrame()
    data['year'] = range(1800, 2000, 10)

    d = []
    for x, y in models_all.items():
        temp = y.most_similar(positive = keyword, topn=1)
        for term, vector in temp:
            d.append(term)

    data['terms'] = d
    st.dataframe(data)

simterms(keyword)

st.subheader('Semantic Change')

def semchange(keyword):

    # get list of all similar words from different periods

    sim_words = []

    for year, model in models_all.items():
        if year in range(1810, 2000, 60):
            tempsim = model.most_similar(keyword, topn=7)
            for term, vector in tempsim:
                sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([models_all[1990][w] for w in sim_words])

    # get vectors of keyword in all periods

    keyword_vectors = np.zeros(shape=(0,300))

    for year, model in models_all.items():
        if year in range(1810, 2000, 60):
            temp_keyword_vector = np.array([model[keyword]])
            keyword_vectors = np.append(keyword_vectors, temp_keyword_vector, axis=0)

    # add keyword vectors from all periods to vectors of similar words 1990

    allvectors = np.append(sim_vectors1990, keyword_vectors, axis=0)

    # "train" PCA model with only similar words
    pca = PCA(n_components=2)
    pca.fit(sim_vectors1990)
    two_dim = pca.transform(allvectors)

    # get labels
    labels = sim_words
    for year, model in models_all.items():
        if year in range(1810, 2000, 60):
            labels.append(keyword + str(year))

    #plot results

    fig, ax = plt.subplots()
    ax.scatter(two_dim[:, 0], two_dim[:, 1])

    texts = [ax.text(x=two_dim[i, 0], y=two_dim[i, 1], s=labels[i]) for i in range(len(sim_words))]
    adjust_text(texts)

    #plot arrow between keywords

    for i in range(-2, -(len(keyword_vectors)+1), -1):
        ax.arrow(two_dim[i,0], two_dim[i,1],
                  two_dim[i+1, 0] - two_dim[i,0], two_dim[i+1, 1] - two_dim[i,1],
                  head_width=0.03, length_includes_head=True)

    st.pyplot(fig)
    plt.close()

semchange(keyword)