import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics

import tensorflow as tf
import pickle
import base64

scaler = StandardScaler()
encoder = LabelEncoder()

@st.cache_data
def load_dataset(synthetic: bool):
    if synthetic:
        df = pd.read_csv('synthetic_dataset.csv', delimiter=',')
        print(df)
    else:
        df = pd.read_csv("winequality-white.csv", delimiter=";")
    g = df.groupby('quality')
    g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

    df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))

    df["quality"] = encoder.fit_transform(df["quality"])
    y = df["quality"]
    x = df.drop("quality", axis=1)
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
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

def RandomForest():
    import streamlit as st
    st.write("By ProxyDreams (Alan) CC0 1.0")
    st.link_button("GitHub", url="https://github.com/dreamsproxy/")
    st.write()
    synth_bool      = st.selectbox("Use Synthetic:", (True, False), index=1)
    random_state    = st.number_input("Random State:", min_value=42, max_value=None)
    run_bool        = st.button("Run!")
    if run_bool:
        x, y = load_dataset(synthetic=synth_bool)
        x_train, x_test, y_train, y_test = split_dataset(x, y, 0.2, random_state)

        model = RandomForestClassifier(random_state=random_state, n_estimators=128, max_features="sqrt", n_jobs=4)

        model.fit(x_train, y_train)
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

def GradientBoosting():
    import streamlit as st
    st.write("By ProxyDreams (Alan) CC0 1.0")
    st.link_button("GitHub", url="https://github.com/dreamsproxy/")
    synth_bool      = st.selectbox("Use Synthetic:", (True, False), index=1)
    random_state    = st.number_input("Random State:", min_value=42, max_value=None)
    run_bool        = st.button("Run!")
    if run_bool:
        x, y = load_dataset(synthetic=synth_bool)
        x_train, x_test, y_train, y_test = split_dataset(x, y, 0.2, random_state)

        model = GradientBoostingClassifier(random_state=random_state, n_estimators=64, max_features="sqrt", n_jobs=4)

        model.fit(x_train, y_train)
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

def NeuralNetwork():
    import streamlit as st
    st.write("By ProxyDreams (Alan) CC0 1.0")
    st.link_button("GitHub", url="https://github.com/dreamsproxy/")
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
                x = tf.keras.layers.Dense(k_size, activation=activ_func)(inputs)
            else:
                x = tf.keras.layers.Dense(k_size, activation=activ_func)(x)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization()(x)
                if isinstance(dropout_ratio, float):
                    x = tf.keras.layers.Dropout(dropout_ratio)(x)

        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
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
        href = f"<a href='data:file/output_model;base64,{b64}'>Download Trained Model .pkl File</a> (right-click and save as &lt;some_name&gt;.pkl)"
        st.markdown(href, unsafe_allow_html=True)

    st.write("## Dense Neural Network Prediction")
    df, x, y = load_dataset(synthetic=True)
    #get_correlation_heatmap(df)
    st.write("Set parameters:")
    dataset_params, model_params = st.columns(2)
    with dataset_params:
        st.write("Data Parameters")
        split_size      = st.number_input("Train Ratio:", min_value=0.5, max_value=0.8, value=0.7)
        random_state    = st.number_input("Random State:", min_value=1, max_value=None, value=42)
        batch_size      = st.selectbox("Batch Size:", (8, 16, 32, 64, 128, 256), index=2)

    with model_params:
        n_layers        = st.number_input("Number of layers:", min_value=2, max_value=5)
        activation_func = st.selectbox("Activation Function:", ("relu", "sigmoid", "exponential"), index=0)
        batch_norm      = st.selectbox("Batch Normalization:", ("True", "False"), index=0)
        dropout_ratio   = st.selectbox("Drop Out Ratio:", ["Do not use dropout"]+list(np.round(np.arange(start=0.1, stop=0.9, step=(0.05)), 2)), index=5)
        epochs          = st.number_input("Epochs:", min_value=1, max_value=500, value=100)

    data_ready_lock = False
    data_ready = st.button("Build Dataset", disabled=data_ready_lock)
    build_lock = False
    build_bool  = st.sidebar.button("Build and Train Model", disabled=build_lock)
    if data_ready:
        data_ready_lock = True
        x_train, x_test, y_train, y_test = split_dataset(x, y, split_size, random_state)
        if build_bool:
            build_lock = False
            with st.spinner("Hold on, model is training..."):
                compiled_model = build_model(11, 7, n_layers, activation_func, batch_norm, dropout_ratio)
                history, model = train(compiled_model, x_train, y_train, batch_size, epochs)
            st.success("Done!")
            eval_result = model.evaluate(x_test, y_test)
            y_pred = model.predict(x_test)
            y_pred = pd.Series([int(np.argmax(row)) for row in y_pred], name='quality')
            #f1_score = sklearn.metrics.f1_score(y_pred, y_test,average="weighted", zero_division='warn')
            c1, c2, c3, c4, c5 = st.columns(5, gap='small')
            with c1:
                st.header("Accuracy")
                st.write(metrics.accuracy_score(y_test, y_pred))
            with c2:
                st.header("Precision")
                st.write(metrics.precision_score(y_test, y_pred, average="weighted", zero_division='warn'))
            with c3:
                st.header("Recall")
                st.write(metrics.recall_score(y_test, y_pred, average="weighted", zero_division='warn'))
            with c4:
                st.header("F1")
                st.write(metrics.f1_score(y_test, y_pred, average="weighted", zero_division='warn'))
            build_lock = True
            st.header("Training log:")
            plot_history(history)

def MainPage():
    import streamlit as st
    st.write("By ProxyDreams (Alan) CC0 1.0")
    st.link_button("GitHub", url="https://github.com/dreamsproxy/")
    st.title("White Wine Quality Prediction")
    st.write("This web app shows 3 different methods to predict wine quality.")
    st.write("The dataset is UNBALANCED")
    forest, gboost, dense = st.columns(3, gap='small')
    with forest:
        st.button("Random Forest Classifier", use_container_width=True, on_click=RandomForest)
    with dense:
        st.button("Dense Neural Network Classifier", use_container_width=True, on_click=NeuralNetwork)
    with gboost:
        st.button("Gradient Boosting Classifier", use_container_width=True, on_click=MainPage)

if __name__ == "__main__":
    st.write("By ProxyDreams (Alan) CC0 1.0")
    st.link_button("GitHub", url="https://github.com/dreamsproxy/")
    st.title("White Wine Quality Prediction")
    st.write("This web app shows 3 different methods to predict wine quality.")
    st.write("The dataset is UNBALANCED")
    st.sidebar.button("Home", use_container_width=True, on_click=MainPage)
    st.sidebar.button("Random Forest Classifier", use_container_width=True, on_click=RandomForest)
    st.sidebar.button("Dense Neural Network Classifier", use_container_width=True, on_click=NeuralNetwork)
    st.sidebar.button("Gradient Boosting Classifier", use_container_width=True, on_click=GradientBoosting)