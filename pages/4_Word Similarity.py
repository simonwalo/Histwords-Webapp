import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

st.subheader('Word Similarity')

keyword1 = st.text_input("Input term 1", "work", key="word1")
keyword1 = keyword1.lower()

keyword2 = st.text_input("Input term 2", "hard", key="word2")
keyword2 = keyword2.lower()

def distchange(keyword1, keyword2):

    d = []

    for year, model in st.session_state['models_all'].items():
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
    ax.set_xticks(range(1810, 2000, 30))

    # show plot
    st.pyplot(fig)
    plt.close()

distchange(keyword1, keyword2)