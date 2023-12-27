import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

encoder = LabelEncoder()
data = pd.read_csv('synthetic_data.csv', delimiter=';')
data = data.sort_values('quality', ascending=False)
print(data['quality'].value_counts())

targets = [3, 9]
n_samples = 15
target_size = 512
for t in targets:
    sample_rows = data.loc[(data['quality'] == t)]
    n_samples = target_size - sample_rows.shape[0]
    gen_samples = dict()
    for col in list(sample_rows.columns):
        if col != 'quality':
            m = sample_rows[col].mean()
            mean_range = (m+m*0.5, m+m*1.5)
            generator = np.random.default_rng()
            gen_samples[col] = pd.Series(generator.uniform(low=m+m*0.5, high=m+m*1.5, size=(n_samples)), name=col)
    gen_samples['quality'] = pd.Series([t for i in range(n_samples)], name='quality')
    gen_samples = pd.DataFrame(gen_samples, columns = data.columns)
    data = pd.concat([data, gen_samples], ignore_index = True)
print(data['quality'].value_counts())
#raise
#g = data.groupby('quality')
#g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
#data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
#print(data['quality'].value_counts())
raise
data.to_csv("synthetic_dataset.csv", sep=",", index=False)
#pd.DataFrame(data).to_csv("9.csv", sep=";")
print(data.shape)
raise
g = data.groupby('quality')
g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
columns = data.columns
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
data = pd.DataFrame(scaler.fit_transform(data), columns = columns)
corr = data.corr()

plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0)
plt.show()