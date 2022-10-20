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

### continue from here ###

    d = []

    for year, model in st.session_state['models_all'].items():
        if year in range(1810, 2000, 30):
            if model[keyword1].all() == st.session_state['models_all'][1810]['biology'].all():
                st.write('Keyword ', keyword1, ' not available for ', year)
            if model[keyword2].all() == st.session_state['models_all'][1810]['biology'].all():
                st.write('Keyword ', keyword2, ' not available for ', year)
            else:
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

    if len(x) < 4:
        st.write('Not enough data points. Please try other keywords.')

    else:

        fun = interp1d(x, y, kind='cubic')

        xnew = np.linspace(x[0], 1990, 100)

        fig, ax = plt.subplots()
        ax.plot(xnew, fun(xnew), '-', x, y, 'o')
        ax.set_xticks(range(1810, 2000, 30))

        # show plot
        st.pyplot(fig)
        fig.clear()
        plt.close(fig)

distchange(keyword1, keyword2)