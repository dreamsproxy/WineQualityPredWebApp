import streamlit as st
st.title("White Wine Quality Prediction", anchor=None)
st.write("White Wine Quality Prediction")

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)