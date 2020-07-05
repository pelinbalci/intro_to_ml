import sys
from common.class_vis import prettyPicture
from common.prep_terrain_data import makeTerrainData
from save_fig import save_fig

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y, features_train, labels_train, features_test, labels_test = makeTerrainData()
clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

# Accuracy

# manual way
sum_false_pred = 0
for i in range(len(labels_test)):
    sum_false_pred += abs(pred[i] - labels_test[i])

total_pred = len(pred)
accuracy_manual = (total_pred - sum_false_pred) / total_pred

# sklearn
accuracy_sklearn = accuracy_score(pred, labels_test)

# sklearn othr way
accuracy_sklearn_2 = clf.score(features_test, labels_test)
print('accuracy_manual {}, sklearn accuracy {}, sklearn score {}'.format(accuracy_manual, accuracy_sklearn, accuracy_sklearn_2))


# Plot the data & classification
fig = prettyPicture(clf, features_test, labels_test)
name = 'im_svm_test'
save_fig(fig, name, 'SVM/')

# support vectors:
print('support vectors:', clf.support_vectors_)

# how many support vectors?
print('support vectors numbers:', clf.n_support_)

# get indices
print('support vector indices:', clf.support_)


