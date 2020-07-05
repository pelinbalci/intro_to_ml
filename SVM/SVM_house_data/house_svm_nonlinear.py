# Ref: https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from save_fig import save_fig

start = time.time()

test_path = 'california_housing_test.csv'
train_path = 'california_housing_train.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_train_shuff = df_train.reindex(np.random.permutation(df_train.index))

df_train_mean = df_train_shuff.mean()
df_train_std = df_train_shuff.std()
df_train_norm = (df_train_shuff - df_train_mean) / df_train_std

df_test_mean = df_test.mean()
df_test_std = df_test.std()
df_test_norm = (df_test - df_test_mean) / df_test_std

threshold = 265000  # This is the 75th percentile for median house values.
df_train_norm["price_is_high"] = (df_train['median_house_value'] > threshold).astype(float)
df_test_norm["price_is_high"] = (df_test['median_house_value'] > threshold).astype(float)

iris = datasets.load_iris()

X_data = df_train_norm[['median_income', 'total_rooms']]
y_data = df_train_norm['price_is_high']

X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

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

#
# models_nonlin = (SVC(kernel='poly', degree=2, gamma='auto', C=0.1),
#                  SVC(kernel='poly', degree=2, gamma='auto', C=100),
#                  SVC(kernel='poly', degree=5, gamma='auto', C=0.1),
#                  SVC(kernel='poly', degree=5, gamma='auto', C=100))
#
# models_nonlin = (clf.fit(X, y) for clf in models_nonlin)
#
# # title for the plots
# titles_nonlin = (
#           'SVC poly degree=2, gamma=auto c=0.1',
#           'SVC poly degree=2, gamma=auto c=100',
#           'SVC poly degree=5, gamma=auto c=0.1',
#           'SVC poly degree=5, gamma=auto c=100',
# )
#
#
# # Set-up 2x2 grid for plotting.
# fig, sub = plt.subplots(2, 2)
# plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
# X0, X1 = X['median_income'], X['total_rooms']
# xx, yy = make_meshgrid(X0, X1)
#
# for clf, title, ax in zip(models_nonlin, titles_nonlin, sub.flatten()):
#     plot_contours(ax, clf, xx, yy,
#                   cmap=plt.cm.coolwarm, alpha=0.8)
#     ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel('Sepal length')
#     ax.set_ylabel('Sepal width')
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.set_title(title, fontsize=8)
#
# fig = plt.gcf()
# name = 'house_svm_nonlinear'
# save_fig(fig, name, 'SVM_house_data/images/')


model_1 = SVC(kernel='poly', degree=2, gamma='auto', C=0.1)
model_1.fit(X, y)
pred_1 = model_1.predict(X_test)
print('kernel=poly, degree=2, gamma=auto, C=0.1 score: ', accuracy_score(y_test, pred_1))

model_2 = SVC(kernel='poly', degree=2, gamma='auto', C=2)
model_2.fit(X, y)
pred_2 = model_2.predict(X_test)
print('kernel=poly, degree=2, gamma=auto, C=2 score: ', accuracy_score(y_test, pred_2))


model_3 = SVC(kernel='poly', degree=5, gamma='auto', C=0.1)
model_3.fit(X, y)
pred_3 = model_3.predict(X_test)
print('kernel=poly, degree=5, gamma=auto, C=0.1 score: ', accuracy_score(y_test, pred_3))

model_4 = SVC(kernel='poly', degree=5, gamma='auto', C=2)
model_4.fit(X, y)
pred_4 = model_4.predict(X_test)
print('kernel=poly, degree=5, gamma=auto, C=2 score: ', accuracy_score(y_test, pred_4))

'''
kernel=poly, degree=2, gamma=auto, C=0.1 score:  0.7850267379679144
kernel=poly, degree=2, gamma=auto, C=2 score:  0.7823529411764706
kernel=poly, degree=5, gamma=auto, C=0.1 score:  0.795187165775401

'''