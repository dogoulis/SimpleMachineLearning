# importing the libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
# importing - split data
data = datasets.load_breast_cancer()
X = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

# applying ensemble method
ens = BaggingClassifier(random_state=42)
ens.fit(x_train, y_train)
y_pred = ens.predict(x_test)

# evaluating
print('Accuracy = ', metrics.accuracy_score(y_test, y_pred))
print('Precision = ', metrics.precision_score(y_test, y_pred))
print('Recall = ', metrics.recall_score(y_test, y_pred))
print('F1 = ', metrics.f1_score(y_test, y_pred))

# applying decission tree
model = RandomForestClassifier(criterion='entropy', n_estimators=200)
model.fit(x_train, y_train)
y_tree_pred = model.predict((x_test))

# evaluating
print('Accuracy = ', metrics.accuracy_score(y_test, y_tree_pred))
print('Precision = ', metrics.precision_score(y_test, y_tree_pred))
print('Recall = ', metrics.recall_score(y_test, y_tree_pred))
print('F1 = ', metrics.f1_score(y_test, y_tree_pred))

# grouped barchart
bar1 = [metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred),
        metrics.f1_score(y_test, y_pred)]
bar2 = [metrics.accuracy_score(y_test, y_tree_pred), metrics.precision_score(y_test, y_tree_pred), metrics.recall_score(y_test, y_tree_pred),
        metrics.f1_score(y_test, y_tree_pred)]

labels = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x-width/2, bar1, width,  label='ensemble')
rects2 = ax.bar(x+width/2, bar2, width, label='Decision Tree')
ax.set_ylabel('Scores')
ax.set_title('Grouped Barchart')
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()