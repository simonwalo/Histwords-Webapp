import streamlit as st
import numpy as np
import pandas as pd
from Main_Page import load_data

st.subheader('Most similar terms')

models_all = load_data()



keyword = st.text_input("Input term", "gay")
keyword = keyword.lower()


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

