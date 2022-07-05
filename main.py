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
fs.ls('bricktamlandstreamlitbucket')

def read_file(filename):
    with fs.open(filename) as f:
        return f.read()

data = pickle.loads(read_file("bricktamlandstreamlitbucket/embeddings1800.pickle"))


mostsim = data.most_similar("work")

st.write(mostsim)

