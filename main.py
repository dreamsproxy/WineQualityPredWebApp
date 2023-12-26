import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential
scaler = StandardScaler()

csv_file = glob.glob("*.csv")
df = pd.read_csv(csv_file[0], delimiter=";")

# Search max correlations
print(df.corr().mean())
print(df.corr().max())
print(df.corr().min())

# Remove max
df = df.drop("total sulfur dioxide", axis=1)
# Check nulls arent present
print(df.isnull().sum())

Y = df['quality']
X = df.drop("quality", axis=1)
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

num_features = X.shape[1]
num_classes = len(Y.unique())

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
inputs = tf.keras.Input(shape=(num_features,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
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
    x_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]
)