from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



#loading and splitting dataset
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')


#changing its form to fit them in the classifier
vectorizer = TfidfVectorizer()
train = vectorizer.fit_transform(newsgroups_train.data)
test = vectorizer.transform(newsgroups_test.data)

#defining the classifier
classifier = MultinomialNB(alpha=10)

#training, predicting, evaluating:
classifier.fit(train, newsgroups_train.target)
predicted = classifier.predict(test)

f1 = metrics.f1_score(newsgroups_test.target, predicted, average='macro')
accuracy = metrics.accuracy_score(newsgroups_test.target, predicted)
recall = metrics.recall_score(newsgroups_test.target, predicted, average='macro')
precision = metrics.precision_score(newsgroups_test.target, predicted, average='macro')
f1 = round(f1,5)
accuracy = round(accuracy,5)
recall = round(recall,5)
precision = round(precision,5)


#building heatmap:


conf_mat = confusion_matrix(newsgroups_test.target, predicted)

fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap='Set3')
plt.title(label=('f1 = '+str(f1), 'accuracy = '+str(accuracy), 'recall = '+str(recall), 'precision = '+str(precision)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
