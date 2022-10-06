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

load_data()