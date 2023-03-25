import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("df_concat.csv")

st.title("DETECTION DES CHURNERS")
st.subheader("Chargement du dataset")

if st.checkbox("Afficher le dataset"):
    st.table(df)