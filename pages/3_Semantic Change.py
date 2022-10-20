import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text

st.subheader('Semantic Change')

keyword = st.text_input("Input term", "gay", key="semkey")
keyword = keyword.lower()

def semchange(keyword):

    if keyword not in st.session_state['models_all'][1810]:
        st.write('Keyword not found in data. Please check for spelling errors.')
        return

    # get list of all similar words from different periods

    sim_words = []

    for year, model in st.session_state['models_all'].items():
        if year in range(1810, 2000, 60):
            if model[keyword].all() == st.session_state['models_all'][1810]['biology'].all():
                st.write('Keyword not available for ', year)
            if model[keyword].all() != st.session_state['models_all'][1810]['biology'].all():
                tempsim = model.most_similar(keyword, topn=7)
                for term, vector in tempsim:
                    sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([st.session_state['models_all'][1990][w] for w in sim_words])

    # get vectors of keyword in all periods

    keyword_vectors = np.zeros(shape=(0,300))

    for year, model in st.session_state['models_all'].items():
        if year in range(1810, 2000, 60):
            if model[keyword].all() != st.session_state['models_all'][1810]['biology'].all():
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
    for year, model in st.session_state['models_all'].items():
        if year in range(1810, 2000, 60):
            if model[keyword].all() != st.session_state['models_all'][1810]['biology'].all():
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
    fig.clear()
    plt.close(fig)

semchange(keyword)
