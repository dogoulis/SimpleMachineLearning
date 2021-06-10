import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler



# importing the data
data = pd.read_csv('creditcard.csv')
# we want to see how imbalanced is our data so we call:
print(data['Class'].value_counts())

# finding the major and the minor values of our target:
data_major = data[data.Class == 0]
data_minor = data[data.Class == 1]

# downsampling majority class i.e. choosing as many samples for class == 0
# as for class == 1

data_major_downsampled = resample(data_major, replace=False, n_samples=492, random_state=42)

#our new data frame:
data_downsampled = pd.concat([data_major_downsampled, data_minor])

# splitting data:
X = data_downsampled.iloc[:, :-1].values
y = data_downsampled.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

# scaling our variables:
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# building the model
# kernel = sigmoid:
sigmoid_svm = SVC(kernel='sigmoid', C=0.1, gamma=0.5, random_state=42)

sigmoid_svm.fit(x_train, y_train)

sigmoid_predicted = sigmoid_svm.predict(x_test)

print('Accuracy in the model with the sigmoid kernel is: ', accuracy_score(y_test, sigmoid_predicted))
print('Recall in the model with the sigmoid kernel is: ', recall_score(y_test, sigmoid_predicted))
print('Precision in the model with the sigmoid kernel is: ', precision_score(y_test, sigmoid_predicted))
print('F1 in the model with the sigmoid kernel is: ', f1_score(y_test, sigmoid_predicted, average='weighted'))

# kernel = rbf:
rbf_svm = SVC(kernel='rbf', C=0.1, gamma=0.3, random_state=42)

rbf_svm.fit(x_train, y_train)

rbf_predicted = rbf_svm.predict(x_test)

print('\nAccuracy in the model with the rbf kernel is: ', accuracy_score(y_test, rbf_predicted))
print('Recall in the model with the rbf kernel is: ', recall_score(y_test, rbf_predicted))
print('Precision in the model with the rbf kernel is: ', precision_score(y_test, rbf_predicted))
print('F1 in the model with the rbf kernel is: ', f1_score(y_test, rbf_predicted))

# kernel = polynomial:
poly_svm = SVC(kernel='poly', C=0.1, gamma=0.2, degree=2, random_state=42)

poly_svm.fit(x_train, y_train)

poly_predicted = poly_svm.predict(x_test)

print('\nAccuracy in the model with the poly kernel is: ', accuracy_score(y_test, poly_predicted))
print('Recall in the model with the poly kernel is: ', recall_score(y_test, poly_predicted))
print('Precision in the model with the poly kernel is: ', precision_score(y_test, poly_predicted))
print('F1 in the model with the poly kernel is: ', f1_score(y_test, poly_predicted))


