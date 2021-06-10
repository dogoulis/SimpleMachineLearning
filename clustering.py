# importing libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, KMeans
# importing - split data
data = datasets.load_breast_cancer()
X = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

# K-MEANS:
kmeans1 = KMeans(n_clusters=2, random_state=42)
kmeans2 = KMeans(n_clusters=3, random_state=42)
# fitting:
kmeans1.fit(x_train)
kmeans2.fit(x_train)
# clustering:
y_predicted1 = kmeans1.predict(x_test)
y_predicted2 = kmeans2.predict(x_test)
# silhouette score:
print(silhouette_score(x_test, y_predicted1, random_state=42))
print(silhouette_score(x_test, y_predicted2, random_state=42))

# Spectral_Clustering:
spec_cluster1 = SpectralClustering(n_clusters=2, random_state=42)
spec_cluster2 = SpectralClustering(n_clusters=3, random_state=42)
# fitting:
y_pred1 = spec_cluster1.fit_predict(x_test)
y_pred2 = spec_cluster2.fit_predict(x_test)
# sihlouette score:
print(silhouette_score(x_test, y_pred1, random_state=42))
print(silhouette_score(x_test, y_pred2, random_state=42))
