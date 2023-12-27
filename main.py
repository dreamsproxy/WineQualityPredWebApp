
def tensorflow_method():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    scaler = StandardScaler()
    encoder = LabelEncoder()
    data = pd.read_csv('winequality-white.csv', delimiter=';')

    corr = data.corr()

    data['quality'] = encoder.fit_transform(data['quality'])
    Y = data['quality']
    X = data.drop('quality', axis=1)

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

    num_features = X.shape[1]
    num_classes = len(Y.unique())

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

    batch_size = 32
    epochs = 100

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]
    )

    fig = px.line(
        history.history,
        y=['loss', 'val_loss'],
        labels={'x': "Epoch", 'y': "Loss"},
        title="Loss Over Time"
    )

    fig.show()

    eval_result = model.evaluate(X_test, y_test)
    y_preds = model.predict(X_test)
    y_preds = pd.Series([int(np.argmax(row)) for row in y_preds], name='quality')
    import sklearn
    sklearn.metrics.f1_score(y_preds, y_test,average="weighted", zero_division='warn')
    

def mutli_model():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from time import process_time

    df_wine= pd.read_csv('./winequality-white.csv', delimiter=";")
    print(df_wine.info())
    print()
    print(f"Unique values:\n{df_wine.nunique()}")

    X = df_wine.drop(['quality'], axis=1)
    y = df_wine['quality']
    # raise
    from sklearn.model_selection import train_test_split
    X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42) # 80-20 split

    # Checking split 
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)
    
    # 1. Using Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics

    rf_classifier = RandomForestClassifier(n_jobs=2, random_state=0)

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")
