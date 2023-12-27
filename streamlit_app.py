import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

import pickle
import base64

scaler = StandardScaler()
encoder = LabelEncoder()

@st.cache_data
def load_dataset():
    df = pd.read_csv('winequality-white.csv', delimiter=';')
    df['quality'] = encoder.fit_transform(df['quality'])
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

def split_dataset(x, y, train_size, state):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,train_size=train_size, random_state=state)
    return x_train, x_test, y_train, y_test

def build_model(num_features, num_classes, n_layers: int, activ_func: str, batch_norm: bool, dropout_ratio):
    inputs = tf.keras.Input(shape=(num_features,))
    
    layer_sizes = []
    b = 32
    for i in range(1, n_layers):
        layer_sizes.append(b)
        b*=2
    layer_sizes.append(layer_sizes[-1])
    layer_sizes.reverse()
    # Construct model:
    for i, k_size in enumerate(layer_sizes):
        if i == 0:
            # First layer -> handle input
            x = tf.keras.layers.Dense(k_size, activation=activ_func)(layer_sizes)
        else:
            x = tf.keras.layers.Dense(k_size, activation=activ_func)(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            if isinstance(dropout_ratio, float):
                x = tf.keras.layers.Dropout(dropout_ratio)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model .pkl File</a> (right-click and save as &lt;some_name&gt;.pkl)'
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    st.title("White Wine Quality Prediction")
    df, x, y = load_dataset()
    #get_correlation_heatmap(df)
    st.write("Train a wine quality prediction model yourself!")
    split_size      = st.sidebar.number_input("Train Ratio:", min_value=0.5, max_value=0.8)
    random_state    = st.sidebar.number_input("Random State:", min_value=1, max_value=None)
    n_layers        = st.sidebar.number_input("Number of layers:", min_value=1, max_value=5)
    activation_func = st.sidebar.selectbox("Activation Function:", ("relu", "sigmoid", "exponential"))
    batch_norm      = st.sidebar.selectbox("Batch Normalization:" ("True", "False"))
    dropout_ratio   = st.sidebar.selectbox("Drop Out Ratio:" ["Do not use dropout"]+list(np.arange(start=0.1, stop=0.9, step=(0.05))))

    batch_size      = st.sidebar.selectbox('Batch Size:', (8, 16, 32, 64, 128, 256))
    epochs          = st.sidebar.number_input("Epochs:", min_value=1, max_value=500)
    
    x_train, x_test, y_train, y_test = split_dataset(x, y, split_size, random_state)
    compiled_model  = st.sidebar.button("Build model.", on_click=build_model, args=(11, 7, n_layers, activation_func, batch_norm, dropout_ratio))
    
    history, model = st.sidebar.button("Start Training Sequence", on_click=train, args=(compiled_model, x_train, y_train))
    st.write("Training log:")
    st.line_chart(history)