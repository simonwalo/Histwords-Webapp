import streamlit as st
import pickle
import s3fs


st.title('Historical Word Embeddings')

st.write("Welcome!")
st.write("This is an interactive web app that allows users to explore how the meaning of words change over time")
st.write("Creator: Simon Walo")
st.write("Data source: https://nlp.stanford.edu/projects/histwords/")


# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.

fs = s3fs.S3FileSystem(anon=False)
fs.ls('bricktamlandstreamlitbucket')

def read_file(filename):
    with fs.open(filename) as f:
        return f.read()


#@st.cache(allow_output_mutation = True)
@st.experimental_memo
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

st.session_state['models_all'] = load_data()
st.write("Data loaded!")
