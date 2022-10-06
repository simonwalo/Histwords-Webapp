import streamlit as st
import pickle
import gensim
import pandas as pd
import sys
import numpy as np
import s3fs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text
from scipy.interpolate import interp1d



st.title('Historical Word Embeddings')

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.

fs = s3fs.S3FileSystem(anon=False)
fs.ls('bricktamlandstreamlitbucket')

def read_file(filename):
    with fs.open(filename) as f:
        return f.read()


@st.cache(allow_output_mutation = True)
def load_data():
    models_all = {
        1810: pickle.loads(read_file("bricktamlandstreamlitbucket/embeddings1810.pickle")),
        1840: pickle.loads(read_file("bricktamlandstreamlitbucket/embeddings1840.pickle")),
        1870: pickle.loads(read_file("bricktamlandstreamlitbucket/embeddings1870.pickle")),
        1900: pickle.loads(read_file("bricktamlandstreamlitbucket/embeddings1900.pickle")),
        1930: pickle.loads(read_file("bricktamlandstreamlitbucket/embeddings1930.pickle")),
        1960: pickle.loads(read_file("bricktamlandstreamlitbucket/embeddings1960.pickle")),
        1990: pickle.loads(read_file("bricktamlandstreamlitbucket/embeddings1990.pickle"))
    }
    return models_all

models_all = load_data()

keyword = st.text_input("Input term", "gay")
keyword = keyword.lower()

st.subheader('Most similar terms')

def similarterms():
    years=[]
    simterms=[]

    for year, model in models_all.items():
        if model[keyword].all() == models_all[1810]['biology'].all():
            st.write('Keyword not available for ', year)
        else:
            years.append(year)
            simterms.append(model.most_similar(keyword))

    simterms2 = []
    for x in simterms:
        for y in x:
            simterms2.append(y[0])

    simterms3 = np.array_split(simterms2, len(models_all))

    simterms4 = []
    for array in simterms3:
        simterms4.append(list(array))

    simterms5 = []
    for x in simterms4:
        simterms5.append((', '.join(x)))

    simtermstable = pd.DataFrame(zip(years, simterms5))
    simtermstable.columns = ["year", "terms"]
    return simtermstable

simtermstable = similarterms()
st.table(simtermstable)




st.subheader('Semantic Change')

def semchange(keyword):

    # get list of all similar words from different periods

    sim_words = []

    for year, model in models_all.items():
        if year in range(1810, 2000, 60):
            if model[keyword].all() == models_all[1810]['biology'].all():
                st.write('Keyword not available for ', year)
            if model[keyword].all() != models_all[1810]['biology'].all():
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
            if model[keyword].all() != models_all[1810]['biology'].all():
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
            if model[keyword].all() != models_all[1810]['biology'].all():
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





st.subheader('Distance between words')

keyword1 = st.text_input("Input term 1", "work")
keyword1 = keyword1.lower()

keyword2 = st.text_input("Input term 2", "hard")
keyword2 = keyword2.lower()

def distchange(keyword1, keyword2):

    d = []

    for year, model in models_all.items():
        if year in range(1810, 2000, 30):
            d.append(
                {
                    "year": year,
                    "similarity": model.n_similarity([keyword1], [keyword2])
                }
            )

    data = pd.DataFrame(d)

    # the trendline
    x = data['year'].tolist()
    y = data['similarity'].tolist()

    fun = interp1d(x, y, kind='cubic')

    xnew = np.linspace(1810, 1990, 100)

    fig, ax = plt.subplots()
    ax.plot(xnew, fun(xnew), '-', x, y, 'o')

    # show plot
    st.pyplot(fig)
    plt.close()

distchange(keyword1, keyword2)