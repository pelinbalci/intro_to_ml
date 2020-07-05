import pandas as pd
from common.class_vis import prettyPicture
from common.prep_terrain_data import makeTerrainData
from save_fig import save_fig

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

# create data
X, y, features_train, labels_train, features_test, labels_test = makeTerrainData()

# define classification
clf = DecisionTreeClassifier(max_depth=2)
name = 'max_depth 2'

# fit train data
clf.fit(features_train, labels_train)

# predict train and test
pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

# calculate accuracy
accuracy_test = accuracy_score(labels_test, pred_test)
accuracy_train = accuracy_score(labels_train, pred_train)
print(name, 'test accuracy {}:, train accuracy {}:'.format(round(accuracy_test, 3), round(accuracy_train, 3)))

# plot the classes
fig = prettyPicture(clf, features_test, labels_test)
save_fig(fig, name, 'Decision_Tree/images')

# confusion matrix
print('confusion_matrix {}:'.format(confusion_matrix(labels_test, pred_test)))
print(classification_report(labels_test,  pred_test, labels=[0, 1]))

# plot roc curve
y_test_pred_prob = clf.predict_proba(features_test)[:, 1]

# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = roc_curve(labels_test, pred_test)

plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
fig = plt.gcf()
name = 'roc_curve'
save_fig(fig, name, 'Decision_Tree/images')

# Find Sensitivity and specificity with threshold value
threshold = 0.5
print('Sensitivity:', tpr[thresholds > threshold][-1])
print('Specificity:', 1 - fpr[thresholds > threshold][-1])

# USE CROSS VALIDATION to get more accurate estimation
print('# USE CROSS VALIDATION to get more accurate estimation')
clf = DecisionTreeClassifier(max_depth=2)
scores = cross_val_score(clf, X, y, cv=5)
print('cv 5 score {}'.format(scores))  # [0.855 0.86  0.86  0.845 0.87 ]
print('cv 5 score mean {}'.format(scores.mean()))


# USE CROSS VALIDATION to find best max_depth
print('USE CROSS VALIDATION to find best max_depth')
all_scores = []
best_score = 0
best_depth = 0
for i in range(1, 10):
  clf = DecisionTreeClassifier(max_depth=i)
  scores = cross_val_score(clf, X, y, cv=5)
  all_scores.append(scores.mean())

  if scores.mean() > best_score:
      best_score = scores.mean()
      best_depth = i

print('best score {}, best depth {}'.format(best_score, best_depth))

# plot different depth scores
plt.figure()
plt.plot(list(range(1, 10)), all_scores)
fig = plt.gcf()
name = 'score for different depths'
save_fig(fig, name, 'Decision_Tree/images')


# Use grid search cv:
print('Gridsearch CV')
clf = DecisionTreeClassifier()

params = dict(max_depth=list(range(1, 10)))
grid = GridSearchCV(clf, params, cv=5)
grid.fit(X, y)

df_grid_results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(df_grid_results)
print('grid.best_score_ {}: '.format(grid.best_score_))
print('grid.best_params_ {}: '.format(grid.best_params_))

# plot gridseacrhcv
plt.figure()
plt.plot(list(range(1, 10)), grid.cv_results_['mean_test_score'])
plt.xlabel('Value of Depth')
plt.ylabel('Cross-Validated Accuracy')
fig = plt.gcf()
name = 'grid_search_cv_depth'
save_fig(fig, name, 'Decision_Tree/images')


# Use gridsearchcv for more than one parameters:
print('Gridsearch fpr more than one param')
from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier()
params = dict(max_depth=list(range(1,10)), min_samples_split=list(range(10,50,10)))
#params = {'max_depth':(list(range(1,10))), 'min_samples_split': [10, 20, 30, 40, 50]}


grid = GridSearchCV(clf, params, cv=5)
grid.fit(X, y)
df_grid_results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(df_grid_results)
print('grid.best_score_ {}: '.format(grid.best_score_))  # grid.best_score_ 0.944:
print('grid.best_params_ {}: '.format(grid.best_params_))  # grid.best_params_ {'max_depth': 5, 'min_samples_split': 10}:


# Use Randomized search:
print('Randomized search')
from sklearn.model_selection import RandomizedSearchCV

clf = DecisionTreeClassifier()
depth_values = list(range(1,10))
min_number_samples = list(range(10,50,10))
param_grid = dict(max_depth=depth_values, min_samples_split=min_number_samples)

rand = RandomizedSearchCV(DecisionTreeClassifier(), param_grid, cv=10, scoring="accuracy", n_iter=10, random_state=5)
rand.fit(X, y)
random_results = pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(random_results)
print('rand.best_score_ {}: '.format(rand.best_score_))  # grid.best_score_ 0.944:
print('rand.best_params_ {}: '.format(rand.best_params_))  # grid.best_params_ {'max_depth': 5, 'min_samples_split': 10}:

"""
cv 5 score [0.855 0.86  0.86  0.845 0.87 ]
cv 5 score mean 0.858

Manual way: 
best score 0.9410000000000001, best depth 5

Gridsearch cv:
   mean_test_score  std_test_score            params
0            0.737        0.029597  {'max_depth': 1}
1            0.858        0.008124  {'max_depth': 2}
2            0.905        0.015166  {'max_depth': 3}
3            0.937        0.012884  {'max_depth': 4}
4            0.947        0.010296  {'max_depth': 5}
5            0.943        0.012884  {'max_depth': 6}
6            0.929        0.013928  {'max_depth': 7}
7            0.928        0.015684  {'max_depth': 8}
8            0.926        0.012410  {'max_depth': 9}
grid.best_score_ 0.9469999999999998: 
grid.best_params_ {'max_depth': 5}: 
"""