# Ref: https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from save_fig import save_fig

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

iris = datasets.load_iris()

X_data = iris['data'][:, 2:]
y_data = iris.target

X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors SVM regularization parameter
models_lin = (SVC(kernel='linear', C=2),
              LinearSVC(C=2, max_iter=10000),
              SVC(kernel='linear', C=0.1),
              LinearSVC(C=0.1, max_iter=10000),
              )

models_lin = (clf.fit(X, y) for clf in models_lin)

# title for the plots
titles_lin = ('SVC with linear kernel c=2',
              'LinearSVC c=2',
              'SVC with linear kernel c=0.1',
              'LinearSVC c=0.1'
              )


# Plot the train set:
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models_lin, titles_lin, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

fig = plt.gcf()
name = 'iris_svm_linear'
save_fig(fig, name, 'SVM/images/')


model_1 = SVC(kernel='linear', C=2)
model_2 = LinearSVC(C=2, max_iter=10000)
model_3 = SVC(kernel='linear', C=0.1)
model_4 = LinearSVC(C=0.1, max_iter=10000)

model_1.fit(X,y)
model_2.fit(X,y)
model_3.fit(X,y)
model_4.fit(X,y)

pred_1 = model_1.predict(X_test)
pred_2 = model_2.predict(X_test)
pred_3 = model_3.predict(X_test)
pred_4 = model_4.predict(X_test)

print('SVC with linear kernel c=2 score:', accuracy_score(y_test, pred_1))
print('LinearSVC c=2 score: ', accuracy_score(y_test, pred_2))
print('SVC with linear kernel c=0.1 score: ', accuracy_score(y_test, pred_3))
print('LinearSVC c=0.1 score: ', accuracy_score(y_test, pred_4))

'''
SVC with linear kernel c=2 score: 0.98
LinearSVC c=2 score:  0.94
SVC with linear kernel c=0.1 score:  1.0
LinearSVC c=0.1 score:  0.82
'''