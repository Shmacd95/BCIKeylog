import numpy as np
import pandas as pd
import tensorflow as tf

X = np.load("eegDat.npy")
y = np.load("eegMrk.npy")

X = X.reshape(len(X),-1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size = 0.20, random_state=11)

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm,metrics

#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)
#print("Accuracy:",metrics.accuracy_score(y_test, clf.predict(X_test)))

clf = svm.LinearSVC(max_iter=50000, random_state=11)
clf.fit(X_train, y_train)
print("F-Score:",metrics.f1_score(y_test, clf.predict(X_test), average='weighted'))
print("Accuracy:",metrics.accuracy_score(y_test, clf.predict(X_test)))
print(metrics.precision_recall_fscore_support(y_test, clf.predict(X_test), average='weighted'))
