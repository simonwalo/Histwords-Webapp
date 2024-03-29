import streamlit as st
import numpy as np
import pandas as pd

st.subheader('Most similar terms')

keyword = st.text_input("Input term", "gay", key="simkey")
keyword = keyword.lower()


def similarterms():

    if keyword not in st.session_state['models_all'][1810]:
        st.write('Keyword not found in data. Please check for spelling errors.')
        return

    years=[]
    simterms=[]

    for year, model in st.session_state['models_all'].items():
        if model[keyword].all() == st.session_state['models_all'][1810]['biology'].all():
            st.write('Keyword not available for ', year)
        else:
            years.append(year)
            simterms.append(model.most_similar(keyword))

    simterms2 = []
    for x in simterms:
        for y in x:
            simterms2.append(y[0])

    simterms3 = np.array_split(simterms2, len(st.session_state['models_all']))

    simterms4 = []
    for array in simterms3:
        simterms4.append(list(array))

    simterms5 = []
    for x in simterms4:
        simterms5.append((', '.join(x)))

    simtermstable = pd.DataFrame(zip(years, simterms5))
    simtermstable.columns = ["year", "terms"]
    st.table(simtermstable)

similarterms()
