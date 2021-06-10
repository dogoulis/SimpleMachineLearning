# importing libraries:
import pandas as pd
import numpy as np
# models:
# ======================================================================================================================
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, SpectralClustering
# ======================================================================================================================
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# importing the dataset:
train_file = pd.read_csv('NSL-KDDTrain.csv')
test_file = pd.read_csv('NSL-KDDTest.csv')
x_train = train_file.iloc[:, :].values
x_train = x_train[:10000, :]
x_test = test_file.iloc[:, :-1].values
y_test = test_file.iloc[:, -1].values


# ======================================================================================================================
# DATA PREPROCESSING
# ======================================================================================================================
# dropping the second and third column
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
x_train = x_train.drop(x_train.columns[[2, 3]], axis=1)
x_test = x_test.drop(x_test.columns[[2, 3]], axis=1)

# perform one hot encoding:
# for the first column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x_train = np.array(ct.fit_transform(x_train))
x_test = np.array(ct.fit_transform(x_test))
print(np.shape(x_train))
print(np.shape(x_test))
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

# data scaling:
print('scaling')
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# pca for dimensionality reduction
print('pca')
pca = PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
# ======================================================================================================================

# ======================================================================================================================
# MODELS FOR CLUSTERING:
# ======================================================================================================================
# Neighbors:
print('Neighbors')
neighbors = KMeans(n_clusters=2, random_state=42)
print('fitting')
neighbors.fit(x_train)
print('first predicting')
neighbors_predicted = neighbors.predict(x_test)

# MiniBatchNeighbors:
print('MiniBatchNeighbors')
mneighbors = MiniBatchKMeans(n_clusters=2, random_state=42)
mneighbors.fit(x_train)
print('second predicting')
mneighbors_predicted = mneighbors.predict(x_test)


# DBSCAN:
print('DBSCAN')
print('visualization for choosing params')
isomap = Isomap(n_components=2, n_neighbors=4)
x_train_emb = isomap.fit_transform(x_train)
emb = SpectralClustering(n_clusters=2, random_state=42)
x_train_spec = emb.fit(x_train)
ax = plt.figure(figsize=(15, 8))
plt.scatter(x_train_emb[:, 0], x_train_emb[:, 1], cmap=plt.cm.Spectral, c=emb.labels_)
plt.title('Spectral Clustering ')
plt.axis('tight')
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=4, metric='euclidean')
print('Dimensionality reduction on x_test')
x_test_spec = emb.fit(x_test)
print('third predicting')
dbscan_predicted = dbscan.fit_predict(x_test_spec)
# ======================================================================================================================

# ======================================================================================================================
# EVALUATION:
# ======================================================================================================================
# evaulation for neighbors:
accuracy_neighbors = accuracy_score(y_test, neighbors_predicted)
print('Accuracy for neighbors is : ', accuracy_neighbors)
# evaulation for mneighbors:
accuracy_mneighbors = accuracy_score(y_test, mneighbors_predicted)
print('Accuracy for mneighbors is : ', accuracy_mneighbors)
# evaluation for dbscan:
accuracy_dbscan = accuracy_score(y_test, dbscan_predicted)
print('Accuracy for dbscan is : ', accuracy_dbscan)
