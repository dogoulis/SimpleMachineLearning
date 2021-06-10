import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
# for the last section, two algorithms that i found interesting:

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier

# ======================================================================================================================

# importing the data
data = pd.read_csv('HTRU_2.csv')
# we want to see how imbalanced is our data so we call:
print(data['0'].value_counts())
# finding the major and the minor values of our target:
data_major = data.loc[data['0'] == 0]
data_minor = data.loc[data['0'] == 1]

# downsampling majority class i.e. choosing as many samples for class == 0
# as for class == 1

data_major_downsampled = resample(data_major, replace=False, n_samples=1639, random_state=42)

# our new data frame:
data_downsampled = pd.concat([data_major_downsampled, data_minor])

# splitting data:
X = data_downsampled.iloc[:, :-1].values
y = data_downsampled.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

# scaling our variables:
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# applying PCA:
pca = PCA(n_components=4)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# ======================================================================================================================
# FEATURE IMPORTANCE
# 1st: feature importance with linear correlation from our dataset:

correlated = data.corr()
correlated = pd.DataFrame(correlated)

# these are the variables with the 5 bigger correlation and also the variable class:
print(correlated.nlargest(5, ['0']))
# the columns are weird-named!

# 2nd: feature importance by implementing linear model (logistic reg)

# building the model

# logistic regression:
model = LogisticRegression()

model.fit(x_train, y_train)

predicted = model.predict(x_test)

# important variables:
results = permutation_importance(model, x_test, y_test, scoring='accuracy')

importance2 = results.importances_mean

print('\nFeature importance with neighboors: ')
for i, j in enumerate(importance2):
    print('Feature %0d:, Score: %.3f' % (i, j))
for v, k in enumerate(sorted(importance2, reverse=True)):
    print('The score of the feature with the %0d biggest importance: %.3f' % (v+1, k))

# building the AUC curve:

# keeping the 'random' classification in variable a:
a = [0 for _ in range(len(y_test))]

b = model.predict_proba(x_test)
b = b[:, 1]
ns_auc = roc_auc_score(y_test, a)
lr_auc = roc_auc_score(y_test, b)

a_falsepr, a_truepr, _ = roc_curve(y_test, a)
b_falsepr, b_truepr, _ = roc_curve(y_test, b)

# building the graph:
plt.plot(a_falsepr, a_truepr)
plt.plot(b_falsepr, b_truepr)
plt.title('AUC CURVE')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# evaluating the model:

print('Accuracy is: ', accuracy_score(y_test, predicted))
print('Recall is: ', recall_score(y_test, predicted))
print('Precision is: ', precision_score(y_test, predicted))
print('F1 is: ', f1_score(y_test, predicted, average='micro'))

# ======================================================================================================================
# applying logistic regression with PCA:

model_pca = LogisticRegression()

model_pca.fit(x_train_pca, y_train)

predicted_pca = model_pca.predict(x_test_pca)

# building the AUC curve:
b_pca = model_pca.predict_proba(x_test_pca)
b_pca = b_pca[:, 1]
ns_auc_pca = roc_auc_score(y_test, a)
lr_auc_pca = roc_auc_score(y_test, b_pca)

a_falsepr_pca, a_truepr_pca, _ = roc_curve(y_test, a)
b_falsepr_pca, b_truepr_pca, _ = roc_curve(y_test, b_pca)

# building the graph:
plt.plot(a_falsepr_pca, a_truepr_pca)
plt.plot(b_falsepr_pca, b_truepr_pca)
plt.title('AUC CURVE with PCA')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# evaluating the model :

print('Accuracy with PCA is: ', accuracy_score(y_test, predicted_pca))
print('Recall with PCA is: ', recall_score(y_test, predicted_pca))
print('Precision with PCA is: ', precision_score(y_test, predicted_pca))
print('F1 with PCA is: ', f1_score(y_test, predicted_pca, average='micro'))


# ======================================================================================================================
# these are also two feature importance algorithm with decision trees that i found interesting:
'''
# feature importance with decisionTreeClass:
model1 = DecisionTreeClassifier()
model1.fit(X, y)

importance1 = model1.feature_importances_
print('Feature importance with trees: ')
for i, j in enumerate(importance1):
    print('Feature: %0d, Score: %.3f' % (i, j))
for i, j in enumerate(sorted(importance1,reverse=True)):
    print('The score of the feature with the %0d biggest importvance: %.3f' % (i+1, j))

# feature importance with nearest neighboors:

model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X, y)

results = permutation_importance(model2, X, y, scoring='accuracy')

importance2 = results.importances_mean

print('\nFeature importance with neighboors: ')
for a, b in enumerate(importance2):
    print('Feature %0d:, Score: %.3f' % (a, b))
for a, b in enumerate(sorted(importance2, reverse=True)):
    print('The score of the feature with the %0d biggest importance: %.3f' % (a+1, b))
 '''