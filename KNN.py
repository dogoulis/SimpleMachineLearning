# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email us: arislaza@csd.auth.gr, ipierros@csd.auth.gr
# =============================================================================

# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd


random.seed = 42
np.random.seed(666)



# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================

#reading the two datasets
data = pd.read_csv('train.csv')


#convet to dataframe
data = pd.DataFrame(data)
#titanic_test = pd.DataFrame(titanic_test)

#dropping useless columns
data1 = data.drop(['Name', 'PassengerId', 'Embarked', 'Ticket', 'Cabin','Parch'], axis='columns')
data_no_impute = data.drop(['Name', 'PassengerId', 'Embarked', 'Ticket', 'Cabin','Parch','Age'], axis='columns')

X = data1.iloc[:, 1:].values
y = data1.iloc[:, 0].values

X_no_impute = data_no_impute.iloc[:, 1:].values
y_no_impute = data_no_impute.iloc[:, 0].values

#one hot encoding for sex and data split

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_no_impute = np.array(ct.fit_transform(X_no_impute))
x_train_no_impute, x_test_no_impute, y_train_no_impute, y_test_no_impute = sklearn.model_selection.train_test_split(X_no_impute, y_no_impute)




# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
#scaling the parameters
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#for no impute data
x_train_no_impute = scaler.fit_transform(x_train_no_impute)
x_test_no_impute = scaler.transform(x_test_no_impute)




# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
imputer = sklearn.impute.KNNImputer(n_neighbors=3)
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)


# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================
Accuracy_imputed = []
Recall_imputed = []
Precision_imputed = []
F1_imputed = []
for n in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=n)
    classifier.fit(x_train_imputed, y_train)
    classifier.predict(x_test_imputed)
    Accuracy_imputed.append(sklearn.metrics.accuracy_score(y_test, classifier.predict(x_test_imputed)))
    Recall_imputed.append(sklearn.metrics.recall_score(y_test, classifier.predict(x_test_imputed)))
    Precision_imputed.append(sklearn.metrics.precision_score(y_test, classifier.predict(x_test_imputed)))
    F1_imputed.append(sklearn.metrics.f1_score(y_test,classifier.predict(x_test_imputed)))
print(Accuracy_imputed)
print(Recall_imputed)
print(Precision_imputed)
print(F1_imputed)
Accuracy = []
Recall = []
Precision = []
F1 = []
for i in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train_no_impute, y_train_no_impute)
    classifier.predict(x_test_no_impute)
    Accuracy.append(sklearn.metrics.accuracy_score(y_test, classifier.predict(x_test_no_impute)))
    Recall.append(sklearn.metrics.recall_score(y_test, classifier.predict(x_test_no_impute)))
    Precision.append(sklearn.metrics.precision_score(y_test, classifier.predict(x_test_no_impute)))
    F1.append(sklearn.metrics.f1_score(y_test_no_impute, classifier.predict(x_test_no_impute)))
print(Accuracy)
print(Recall)
print(Precision)
print(F1)

# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================
plt.title('k-Nearest Neighbors (Weights = unifrom, Metric = Minkowski, p = 2)')
plt.plot(F1_imputed, label='with impute')
plt.plot(F1, label='without impute')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('F1')
plt.show()

#for the no_impute data, i dropped age, cause this algorithm doesnt work with nan values