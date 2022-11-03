import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

st.subheader('Word Similarity')

col1, col2 = st.columns(2)

with col1:
    keyword1 = st.text_input("Input term A1", "work", key="word1")
    keyword1 = keyword1.lower()

    keyword3 = st.text_input("Input term B1", "test", key="word3")
    keyword3 = keyword3.lower()

with col2:
    keyword2 = st.text_input("Input term A2", "hard", key="word2")
    keyword2 = keyword2.lower()

    keyword4 = st.text_input("Input term B2", "hello", key="word4")
    keyword4 = keyword4.lower()

def distchange(keyword1, keyword2):

    if keyword1 not in st.session_state['models_all'][1810]:
        st.write('Input term A1 not found in data. Please check for spelling errors.')
        return
    if keyword2 not in st.session_state['models_all'][1810]:
        st.write('Input term A2 not found in data. Please check for spelling errors.')
        return
    if keyword3 not in st.session_state['models_all'][1810]:
        st.write('Input term B1 not found in data. Please check for spelling errors.')
        return
    if keyword4 not in st.session_state['models_all'][1810]:
        st.write('Input term B2 not found in data. Please check for spelling errors.')
        return


    d1 = []
    d2 = []

    for year, model in st.session_state['models_all'].items():
        if year in range(1810, 2000, 30):
            if model[keyword1].all() == st.session_state['models_all'][1810]['biology'].all():
                st.write('Keyword ', keyword1, ' not available for ', year)
            if model[keyword2].all() == st.session_state['models_all'][1810]['biology'].all():
                st.write('Keyword ', keyword2, ' not available for ', year)
            else:
                d1.append(
                    {
                        "year": year,
                        "similarity": model.n_similarity([keyword1], [keyword2])
                    }
                )

    for year, model in st.session_state['models_all'].items():
        if year in range(1810, 2000, 30):
            if model[keyword1].all() == st.session_state['models_all'][1810]['biology'].all():
                st.write('Keyword ', keyword3, ' not available for ', year)
            if model[keyword2].all() == st.session_state['models_all'][1810]['biology'].all():
                st.write('Keyword ', keyword4, ' not available for ', year)
            else:
                d2.append(
                    {
                        "year": year,
                        "similarity": model.n_similarity([keyword3], [keyword4])
                    }
                )

    data1 = pd.DataFrame(d1)
    data2 = pd.DataFrame(d2)


    # the trendline
    x1 = data1['year'].tolist()
    x2 = data2['year'].tolist()

    y1 = data1['similarity'].tolist()
    y2 = data2['similarity'].tolist()


    if len(x1) < 4 or len(x2) < 4:
        st.write('Not enough data points. Please try other keywords.')

    else:

        fun1 = interp1d(x1, y1, kind='cubic')
        fun2 = interp1d(x2, y2, kind='cubic')


        x1new = np.linspace(x1[0], 1990, 100)
        x2new = np.linspace(x2[0], 1990, 100)


        fig, ax = plt.subplots()
        ax.plot(x1new, fun1(x1new), '-', label=(keyword1, keyword2))
        ax.plot(x1, y1, 'o')
        ax.plot(x2new, fun2(x2new), '-', label=(keyword3, keyword4))
        ax.plot(x2, y2, 'o')
        ax.legend()
        ax.set_xticks(range(1810, 2000, 30))

        # show plot
        plt.xlabel("Year")
        plt.ylabel("Cosine Similarity")
        st.pyplot(fig)
        fig.clear()
        plt.close(fig)

distchange(keyword1, keyword2)