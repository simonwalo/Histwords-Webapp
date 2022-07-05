import streamlit as st
import pickle
import gensim
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from adjustText import adjust_text
import s3fs


st.title('Historical Word Embeddings')

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)

@st.experimental_memo(ttl=600)
def read_file(filename):
    with fs.open(filename) as f:
        return f.read().decode("utf-8")

data = read_file("bricktamlandstreamlitbucket/embeddings1800.pickle")

st.write(data.most_similar("work"))




