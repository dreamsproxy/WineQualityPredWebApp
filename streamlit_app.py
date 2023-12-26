import streamlit as st
import pandas as pd
import plotly.express as px
@st.cache_data
def load_dataset():
    df = pd.read_csv('winequality-white.csv', delimiter=';')
    return df
df = load_dataset()
def get_correlation_heatmap(df):
    correlation = df.corr()

    fig = px.imshow(correlation, text_auto=True, width=800, height=800, aspect="equal")
    st.plotly_chart(fig, theme="streamlit")

st.title("<center>White Wine Quality Prediction</center>")

get_correlation_heatmap(df)

st.line_chart()