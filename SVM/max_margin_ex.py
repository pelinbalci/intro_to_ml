# Ref: https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

from save_fig import save_fig


# we create 40 separable points
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# fit the model, don't regularize for illustration purposes
clf_1 = SVC(kernel='linear', C=1000)
clf_1.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_1.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf_1.support_vectors_[:, 0], clf_1.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
fig = plt.gcf()
name = 'max_margin_high_c'
save_fig(fig, name, 'SVM/images/')


# fit the model, don't regularize for illustration purposes
clf_2 = SVC(kernel='linear', C=0.1)
clf_2.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_2.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf_2.support_vectors_[:, 0], clf_2.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
fig_2 = plt.gcf()
name = 'max_margin_low_c'
save_fig(fig_2, name, 'SVM/images/')
print('done')