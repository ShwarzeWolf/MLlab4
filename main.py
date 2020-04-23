import os
import pandas as pd
dataFolder = "C://Users//1358365//PycharmProjects//lab4//messages"
folders = list(os.walk(dataFolder))

rawData = []

for folder in folders[0][1]:
    messages = list(os.walk(dataFolder + '//' + folder))

    for message in messages[0][2]:
        f = open(dataFolder + '//' + folder + '//' + message)

        if (message.find("spmsg") != -1):
            label = "spam"
            labelNum = 1
        else:
            label = "legit"
            labelNum = 0

        text = f.read().replace("Subject:", '')

        rawData.append([label, text, labelNum])
        f.close()

data = pd.DataFrame(rawData, columns = ['label', 'text', 'labelNum'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = data.text
Y = data.labelNum

X_train = X.iloc[:981]
Y_train = Y.iloc[:981]

X_test = X.iloc[981:]
Y_test = Y.iloc[981:]

vectorizer = CountVectorizer()

counts = vectorizer.fit_transform(X_train.values)
X_test = vectorizer.transform(X_test.values)

targets = Y_train.values

classifier = MultinomialNB().fit(counts, targets)
predictions = classifier.predict(X_test)

from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt

fpr, tpr, threshold = roc_curve(Y_test, predictions)

print(accuracy_score(Y_test, predictions))

plt.plot(fpr, tpr)
plt.show()

scores = []
steps = []

result = 0
Y_test = Y_test.to_numpy()

for i in range(predictions.size):
    if predictions[i] == 1 & Y_test[i] == 0:
        result = result + 1

step = 10.0
iterationNumber = 0

while result != 0:
    classifier = MultinomialNB(class_prior=[1 / step, 1 -  (1/ step)])
    classifier.fit(counts, targets)

    predictions = classifier.predict(X_test)
    result = 0

    for i in range(predictions.size):
        if predictions[i] == 1 & Y_test[i] == 0:
            result = result + 1

    print(result)
    scores.append(accuracy_score(Y_test, predictions))
    steps.append(iterationNumber)

    step = step * 10.0

    iterationNumber += 1
    if iterationNumber == 300:
        break

plt.plot(steps, scores)
plt.show()
