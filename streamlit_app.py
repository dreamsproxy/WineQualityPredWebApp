import streamlit as st
import pandas as pd
import plotly.express as px
@st.cache_data
def load_dataset():
    df = pd.read_csv('winequality-white.csv', delimiter=';')
    return df

st.title("White Wine Quality Prediction")
st.write("White Wine Quality Prediction")
st.line_chart()

def get_correlation_heatmap():

    z = [[.1, .3, .5, .7, .9],
         [1, .8, .6, .4, .2],
         [.2, 0, .5, .7, .9],
         [.9, .8, .4, .2, 0],
         [.3, .4, .5, .7, 1]]

    fig = px.imshow(z, text_auto=True)
    st.plotly_chart(fig, theme="streamlit")

get_correlation_heatmap()
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)