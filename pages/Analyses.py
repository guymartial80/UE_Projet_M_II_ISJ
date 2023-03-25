import streamlit as st
from matplotlib import pyplot as plt
from plotly import graph_objs as go

st.title("DETECTION DES CHURNERS")
st.subheader("Analyses exploratoires")

graph = st.selectbox("Quel choix de type de graphique ? ", ["Non-Interactive", "Interactive"])

if graph == "Non-Interactive":
    plt.figure(figsize=(15,10))
if graph == "Interactive":
    pass