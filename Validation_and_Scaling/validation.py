#!/usr/bin/python

"""
PLEASE NOTE:
The api of train_test_split changed and moved from sklearn.cross_validation to
sklearn.model_selection(version update from 0.17 to 0.18)

The correct documentation for this quiz is here:
http://scikit-learn.org/0.17/modules/cross_validation.html
"""
import pandas as pd

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from common.prep_terrain_data import makeTerrainData

# create data
features, labels, features_train_data, labels_train_data, features_test_data, labels_test_data = makeTerrainData()


# # load data
# iris = datasets.load_iris()
# features = iris.data
# labels = iris.target

# split data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.4, random_state=0)

# SVC classifier
clf = SVC(kernel="linear", C=1.)
clf.fit(features_train, labels_train)
print(clf.score(features_test, labels_test))

# cross validation - no more fit
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, features, labels, cv=5)
mean_cross_val_score = scores.mean()
print('cross val score', mean_cross_val_score)

# grid search cv for finding params  -- use fit.
clf = DecisionTreeClassifier()
params = {'max_depth':(list(range(1,10))), 'min_samples_split': [10, 20, 30, 40, 50]}
grid_search = GridSearchCV(clf, params, cv=5)
grid_search.fit(features, labels)
best_score = grid_search.best_score_
best_params = grid_search.best_params_
results = grid_search.cv_results_

'results is a dictionary. We re interested in mean_test score and related params.'
df_results = pd.DataFrame(results)
df_results = df_results[['params', 'mean_test_score']]

print('best score ', best_score)
print('best params', best_params)
print('results ', df_results)

# randomizes search cv
clf = DecisionTreeClassifier()
params = {'max_depth':(list(range(1,10))), 'min_samples_split': [10, 20, 30, 40, 50]}
random_search = RandomizedSearchCV(clf, params, n_iter=5, cv=5)
random_search.fit(features, labels)
best_score = random_search.best_score_
best_params = random_search.best_params_
results = random_search.cv_results_
df_results = pd.DataFrame(results)[['params', 'mean_test_score']]

print('best score ', best_score)
print('best params', best_params)
print('results ', df_results)

# randomizes search cv
clf = SVC()
params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']}
random_search = RandomizedSearchCV(clf, params, n_iter=5, cv=5)
random_search.fit(features, labels)
best_score = random_search.best_score_
best_params = random_search.best_params_
results = random_search.cv_results_
df_results = pd.DataFrame(results)[['params', 'mean_test_score']]

print('best score ', best_score)
print('best params', best_params)
print('results ', df_results)