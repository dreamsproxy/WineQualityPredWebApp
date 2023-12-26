import streamlit as st
st.title("White Wine Quality Prediction")
st.write("White Wine Quality Prediction")
st.line_chart()

def get_correlation_heatmap():
    import plotly.express as px

    z = [[.1, .3, .5, .7, .9],
         [1, .8, .6, .4, .2],
         [.2, 0, .5, .7, .9],
         [.9, .8, .4, .2, 0],
         [.3, .4, .5, .7, 1]]

    fig = px.imshow(z, text_auto=True)

    tab1, tab2 = st.tabs(["Data Correlation"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)