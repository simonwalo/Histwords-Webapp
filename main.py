import streamlit as st
import pickle
import s3fs
from gensim.models import KeyedVectors


st.title('Historical Word Embeddings')

st.write("Welcome!")
st.write("This is an interactive web app that allows users to explore how the meaning of words change over time. Use the sidebar on the left to navigate.")
st.write("Please note: The app is still under development and things might not always work properly.")
st.write("Creator: Simon Walo")
st.write("Data source: https://nlp.stanford.edu/projects/histwords/ (All English (1800s-1990s))")
st.write("Please wait while the data is loading:")

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.

fs = s3fs.S3FileSystem(anon=False)
fs.ls('bricktamlandstreamlitbucket')

def read_file(filename):
    with fs.open(filename) as f:
        return f.read()


@st.experimental_memo
def load_data():
    models_all = {
        1810: KeyedVectors.load(read_file("bricktamlandstreamlitbucket/vectors1810.kv")),
        1840: KeyedVectors.load(read_file("bricktamlandstreamlitbucket/vectors1840.kv")),
        1870: KeyedVectors.load(read_file("bricktamlandstreamlitbucket/vectors1870.kv")),
        1900: KeyedVectors.load(read_file("bricktamlandstreamlitbucket/vectors1900.kv")),
        1930: KeyedVectors.load(read_file("bricktamlandstreamlitbucket/vectors1930.kv")),
        1960: KeyedVectors.load(read_file("bricktamlandstreamlitbucket/vectors1960.kv")),
        1990: KeyedVectors.load(read_file("bricktamlandstreamlitbucket/vectors1990.kv"))
    }
    return models_all




st.session_state['models_all'] = load_data()
st.write("Data loaded!")

data = load_data()
