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
    fig = px.imshow(correlation, title = "Correlation Heatmap",
                    text_auto=True,
                    width=800, height=800, aspect="equal")
    st.plotly_chart(fig, theme="streamlit")

if __name__ == "__main__":
    st.title("White Wine Quality Prediction")
    st.write()
    get_correlation_heatmap(df)

    st.line_chart()