
def tensorflow_method():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    scaler = MinMaxScaler()
    encoder = LabelEncoder()
    #data = pd.read_csv('winequality-white.csv', delimiter=';')
    data = pd.read_csv('synthetic_dataset.csv', delimiter=',')

    g = data.groupby('quality')
    g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

    data = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)).reset_index(drop=True)
    data = data.drop(columns=['density', 'chlorides'])
    #corr = data.corr()
    #sns.heatmap(corr, cmap="viridis", annot=True)
    #plt.show()

    data['quality'] = encoder.fit_transform(data['quality'])
    Y = data['quality']
    X = data.drop('quality', axis=1)
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=34, shuffle=True)

    num_features = X.shape[1]
    num_classes = len(Y.unique())

    inputs = tf.keras.Input(shape=(num_features,))
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.8)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    adam_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    METRICS = [
      'accuracy',
      tf.keras.metrics.SparseTopKCategoricalAccuracy(name='precision'),
    ]

    model.compile(
        optimizer=adam_opt,
        loss=['sparse_categorical_crossentropy'],
        metrics=METRICS
    )

    batch_size = 512
    epochs = 512

    history = model.fit(
        X_train,
        y_train,
        validation_data = (X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[]
    )
    model.save("./latest.keras")
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
    sklearn.metrics.f1_score(y_preds, y_test, average="weighted", zero_division='warn')

tensorflow_method()

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
    x_train, x_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Checking split 
    print('X_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', x_test.shape)
    print('y_test:', y_test.shape)
    
    # 1. Using Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    import sklearn
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    rfc = RandomForestClassifier(random_state=0)
    param_grid = {
        "max_depth": [100, None],  
        "max_features": ["sqrt", "log"],  
        "n_estimators": [128, 256, 512]}
    model = GridSearchCV(rfc, param_grid=param_grid, cv = 5, n_jobs=-1)
    print("Search done.")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")

#mutli_model()