import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

scaler = StandardScaler()
encoder = LabelEncoder()

@st.cache_data
def load_dataset(synthetic: bool):
    if synthetic:
        df = pd.read_csv('synthetic_dataset.csv', delimiter=',')
        print(df)
    else:
        df = pd.read_csv("winequality-white.csv", delimiter=";")
    
    #df["quality"] = encoder.fit_transform(df["quality"])
    y = df["quality"]
    x = df.drop("quality", axis=1)
    #x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    return  x, y

def get_correlation_heatmap(df):
    correlation = df.corr()
    fig = px.imshow(correlation, title = "Correlation Heatmap",
                    text_auto=True,
                    width=800, height=800, aspect="equal")
    st.plotly_chart(fig, theme="streamlit")

def plot_history(history):
    fig = px.line(
        history.history,
        y=["loss", "val_loss", "accuracy", "val_accuracy"],
        labels={"x": "Epoch", "y": "Loss"},
        title="Training Log",
        width=1080,
        height=800
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def split_dataset(x, y, train_size, state):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,train_size=train_size, random_state=state)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    st.title("White Wine Quality Prediction")
    st.write("Train a wine quality prediction model yourself!")
    synth_bool      = st.selectbox("Use Synthetic:", (True, False), index=1)
    random_state    = st.number_input("Random State:", min_value=1, max_value=None)
    
    x, y = load_dataset(synthetic=synth_bool)
    x_train, x_test, y_train, y_test = split_dataset(x, y, 0.2, random_state)

    rfc = RandomForestClassifier(
        n_estimators=1024,
        #criterion="entropy",
        n_jobs=-1,
        random_state=random_state)
    
    param_grid = {
        "max_features": ["sqrt", "log"],  
        "min_samples_split": [1, 3, 5, 10],  
        "min_samples_leaf": [1, 2],  
        "n_estimators": [32, 64,4128, 256, 512, 1024]}
    
    model = GridSearchCV(rfc, param_grid=param_grid, cv = 5, n_jobs=4)
    print("Search done.")
    model.fit(x_train, y_train)
    print("Fit done.")
    y_pred = model.predict(x_test)
    c1, c2, c3, c4 = st.columns(4, gap='small')
    with c1:
        st.header("Accuracy")
        st.write(metrics.accuracy_score(y_test, y_pred))
    with c2:
        st.header("Precision")
        st.write(metrics.precision_score(y_test, y_pred, average="weighted"))
    with c3:
        st.header("Recall")
        st.write(metrics.recall_score(y_test, y_pred, average="weighted"))
    with c4:
        st.header("F1")
        st.write(metrics.f1_score(y_test, y_pred, average="weighted"))