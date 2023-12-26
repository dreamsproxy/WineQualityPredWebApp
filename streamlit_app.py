import streamlit as st

st.write("White Wine Quality Prediction")

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)