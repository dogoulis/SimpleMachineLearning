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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# importing the dataset:
train_file = pd.read_csv('NSL-KDDTrain.csv')
test_file = pd.read_csv('NSL-KDDTest.csv')
x_train = train_file.iloc[:, :].values
#x_train = x_train[:10000, :]
x_test = test_file.iloc[:, :-1].values
y_test = test_file.iloc[:, -1].values

# ======================================================================================================================
# DATA PREPROCESSING
# ======================================================================================================================
# dropping the third and fourth columns
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

# perform label encoding:
encoder = LabelEncoder()
y_test = encoder.fit_transform(y_test)

# data scaling:
print('scaling')
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# pca for dimensionality reduction
print('pca')
pca = PCA(n_components = 0.9)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
# ======================================================================================================================

# ======================================================================================================================
# clustering in order to label them:
# ======================================================================================================================
# Neighbors:
print('Neighbors')
neighbors = KMeans(n_clusters=2, random_state=42)
print('fitting')
neighbors.fit(x_train)
y_cluster = neighbors.labels_

print(np.shape(y_cluster))
print(np.shape(x_train))

# logistic regression
log = LogisticRegression()
log.fit(x_train,y_cluster)
y_log_pred = log.predict(x_test)

# 5-nn
nn = KNeighborsClassifier(n_neighbors=5, weights='distance')
nn.fit(x_train, y_cluster)
y_nn_pred = nn.predict(x_test)

# decission tree
tree = DecisionTreeClassifier()
tree.fit(x_train, y_cluster)
y_tree_pred = tree.predict(x_test)

# naive bayes
GB = GaussianNB()
GB.fit(x_train, y_cluster)
y_gb_pred = GB.predict(x_test)

# evaluating:
print('Accuracy with log reg is: ', accuracy_score(y_test, y_log_pred))
print('Accuracy with 5-nn is: ', accuracy_score(y_test, y_nn_pred))
print('Accuracy with tree is: ', accuracy_score(y_test, y_tree_pred))
print('Accuracy with naive bayes is: ', accuracy_score(y_test, y_gb_pred))