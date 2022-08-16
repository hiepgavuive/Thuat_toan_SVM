import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split



digitel = datasets.load_digits()

# print(digitel.DESCR)

plt.gray()
plt.matshow(digitel.images[9])
plt.show()

n = len(digitel.images)
data = digitel.images.reshape((n, -1))

X_train, X_test, Y_train, Y_test = train_test_split(data, digitel.target, test_size=0.25, shuffle=True)

classifier = svm.SVC()
classifier.fit(X_train, Y_train)

predicted = classifier.predict(X_test)

print(metrics.classification_report(Y_test, predicted))





