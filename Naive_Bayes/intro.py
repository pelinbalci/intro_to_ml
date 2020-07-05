# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

import numpy as np
from sklearn.naive_bayes import GaussianNB

# Create Data #

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])


# Call and fit GaussianNB #

clf = GaussianNB()
clf.fit(X, Y)
GaussianNB()
print(clf.predict([[-0.8, -1]]))

# Call and fit partial GaussianNB #
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
GaussianNB()
print(clf_pf.predict([[-0.8, -1]]))