import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
scaler = StandardScaler()
encoder = LabelEncoder()

@st.cache_data
def load_dataset():
    df = pd.read_csv('winequality-white.csv', delimiter=';')
    df['quality'] = scaler.fit_transform(df['quality'])
    y = df['quality']
    x = df.drop('quality', axis=1)
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    return df, x, y

def get_correlation_heatmap(df):
    correlation = df.corr()
    fig = px.imshow(correlation, title = "Correlation Heatmap",
                    text_auto=True,
                    width=800, height=800, aspect="equal")
    st.plotly_chart(fig, theme="streamlit")

def split_dataset(x, y, state):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,train_size=0.7, random_state=state)
    return x_train, x_test, y_train, y_test

def build_model(num_features, num_classes):
    inputs = tf.keras.Input(shape=(num_features,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@st.cache_data
def train(model, x_train, y_train, batch_size = 32, epochs = 100):
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]
    )
    return history, model

if __name__ == "__main__":
    st.title("White Wine Quality Prediction")
    df, x, y = load_dataset()
    #get_correlation_heatmap(df)
    st.write("Train a wine quality prediction model yourself!")
    add_selectbox = st.sidebar.selectbox(
        'Batch Size:',
        (8, 16, 32, 64, 128, 256)
    )
    x_train, x_test, y_train, y_test = split_dataset(x, y)
    compiled_model = build_model(11, 7)
    train(compiled_model)
    st.line_chart()