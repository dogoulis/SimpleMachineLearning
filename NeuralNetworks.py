from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ======================================================================================================================

# importing the dataset:

data  = datasets.load_breast_cancer()
X = data.data
y = data.target

# splitting the dataset:

x_train, x_test, y_train, y_test = train_test_split(X,y,random_state=42)

# scaling the data:

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# training the model:

# for 10 hidden layers:
recall_list = []
precision_list = []
accuracy_list = []
f1_list = []
for i in ['relu', 'tanh']:
    for j in [0.0001, 0.00001]:
        for k in ['sgd', 'adam', 'lbfgs']:
            classifier = MLPClassifier(max_iter=100, activation=i, tol=j, solver=k, hidden_layer_sizes=10)
            classifier.fit(x_train, y_train)
            predicted = classifier.predict(x_test)
            recall_list.append(metrics.recall_score(y_test, predicted))
            precision_list.append(metrics.precision_score(y_test, predicted))
            accuracy_list.append(metrics.accuracy_score(y_test, predicted))
            f1_list.append(metrics.f1_score(y_test, predicted))

# printing the results
print(recall_list)
print(precision_list)
print(accuracy_list)
print(f1_list)

# for 20 hidden layers:
recall_list = []
precision_list = []
accuracy_list = []
f1_list = []
for i in ['relu', 'tanh']:
    for j in [0.0001, 0.00001]:
        for k in ['sgd', 'adam', 'lbfgs']:
            classifier = MLPClassifier(max_iter=100, activation=i, tol=j, solver=k, hidden_layer_sizes=20)
            classifier.fit(x_train, y_train)
            predicted = classifier.predict(x_test)
            recall_list.append(metrics.recall_score(y_test, predicted))
            precision_list.append(metrics.precision_score(y_test, predicted))
            accuracy_list.append(metrics.accuracy_score(y_test, predicted))
            f1_list.append(metrics.f1_score(y_test, predicted))

# printing the results
print(recall_list)
print(precision_list)
print(accuracy_list)
print(f1_list)

# for 50 hidden layers:
recall_list = []
precision_list = []
accuracy_list = []
f1_list = []
for i in ['relu', 'tanh']:
    for j in [0.0001, 0.00001]:
        for k in ['sgd', 'adam', 'lbfgs']:
            classifier = MLPClassifier(max_iter=100, activation=i, tol=j, solver=k, hidden_layer_sizes=50)
            classifier.fit(x_train, y_train)
            predicted = classifier.predict(x_test)
            recall_list.append(metrics.recall_score(y_test, predicted))
            precision_list.append(metrics.precision_score(y_test, predicted))
            accuracy_list.append(metrics.accuracy_score(y_test, predicted))
            f1_list.append(metrics.f1_score(y_test, predicted))

# printing the results
print(recall_list)
print(precision_list)
print(accuracy_list)
print(f1_list)

# last case:
classifier = MLPClassifier(max_iter=100, activation='relu', tol=0.00001, solver='adam', hidden_layer_sizes=(50,50,50))
classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)
metrics.recall_score(y_test, predicted)
metrics.precision_score(y_test, predicted)
metrics.accuracy_score(y_test, predicted)
metrics.f1_score(y_test, predicted)

# printing the results
print(metrics.recall_score(y_test, predicted))
print(metrics.precision_score(y_test, predicted))
print(metrics.accuracy_score(y_test, predicted))
print(metrics.f1_score(y_test, predicted))



