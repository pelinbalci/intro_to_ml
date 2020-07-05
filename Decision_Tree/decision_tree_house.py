import pandas as pd
import numpy as np
from save_fig import save_fig

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
# try:
#     from matplotlib import pyplot as plt
# except ImportError:
#     import matplotlib
#     matplotlib.use("TkAgg", force=True)
#     import matplotlib.pyplot as plt

# collect data
df_train = pd.read_csv('california_housing_train.csv')
df_test = pd.read_csv('california_housing_test.csv')

# Analyze target
_, axs = plt.subplots(1, sharex='col', figsize=(14, 10))
axs.hist(df_train['median_house_value'])
fig = plt.gcf()
name = 'house_target_hist'
save_fig(fig, name, 'Decision_Tree/images')

# Analyze features
attributes = ["housing_median_age", "population", "households", "median_income", "median_house_value"]
scatter_matrix(df_train[attributes], figsize=(17, 15))
plt.savefig(r"scatter_all_features.png")


# create classes for target value
df_test["house_value"] = np.where(df_test["median_house_value"] >= 150000, 1, 0)
df_train["house_value"] = np.where(df_train["median_house_value"] >= 150000, 1, 0)

# train and test data
feature_train = df_train[["housing_median_age", "population", "households", "median_income"]]
target_train = df_train["house_value"]
feature_test = df_test[["housing_median_age", "population", "households", "median_income"]]
target_test = df_test["house_value"]

# merge train and test data
feature = pd.concat([feature_train, feature_test])
target = pd.concat([target_train, target_test])

# simple classifier
clf = DecisionTreeClassifier()
clf.fit(feature_train, target_train)
pred_train = clf.predict(feature_train)
pred_test = clf.predict(feature_test)

acc_score_train = accuracy_score(target_train, pred_train)
acc_score_test = accuracy_score(target_test, pred_test)

print('acc_score_train {}, acc_score_test {}'.format(acc_score_train, acc_score_test))
"""acc_score_train 1.0  acc_score_test 0.72"""

print(confusion_matrix(target_test, pred_test))
"""[[ 706  390]
 [ 422 1482]]"""

# Cross validation score don't use the splitted train and test values. it randomly splits train and test 5 times.
cv_score = cross_val_score(clf, feature, target, cv=5)
print('cross validation score {}: ', cv_score.mean())
""" 0.7116 """

fpr, tpr, thresholds = roc_curve(target_test, pred_test)
auc = metrics.auc(fpr, tpr)
print('auc {}'.format(auc))

sensitivity = tpr[1]
specificity = 1 - fpr[1]
print('sensitivity {}: , specificity {}:'.format(sensitivity, specificity))
"""sensitivity: 0.7783, specificity: 0.6441"""

classification_report = classification_report(target_test, pred_test, labels=[0,1])
print(classification_report)
"""
              precision    recall  f1-score   support

           0       0.63      0.64      0.63      1096
           1       0.79      0.78      0.78      1904

    accuracy                           0.73      3000
   macro avg       0.71      0.71      0.71      3000
weighted avg       0.73      0.73      0.73      3000
"""

# plot roc curve
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
fig = plt.gcf()
name = 'roc_curve_house_simple'
save_fig(fig, name, 'Decision_Tree/images')


#######
# IMPROVEMENT #
#######

# Manual way:

best_depth = 0
best_score = 0
for i in range(1, 10):
    clf = DecisionTreeClassifier(max_depth=i)
    mean_cv_score = cross_val_score(clf, feature, target, cv=5).mean()
    if mean_cv_score > best_score:
        best_score = mean_cv_score
        best_depth = i

print('best depth {}, best score {}'.format(best_depth, best_score))
"""7, 0.7696"""


# Use grid search cv:
clf = DecisionTreeClassifier()
params_dict = dict(max_depth=list(range(1, 10)))
grid = GridSearchCV(clf, params_dict, cv=5)
grid.fit(feature, target)
print("grid best params: {}, grid best score: {}".format(grid.best_params_, grid.best_score_))
"""{'max_depth': 7}, 0.7696"""

# Use grid seacrh cv for more than one param
clf = DecisionTreeClassifier()
params_dict = dict(max_depth=list(range(1, 10)), min_samples_split=list(range(10, 50, 10)))
grid_all = GridSearchCV(clf, params_dict, cv=5)
grid_all.fit(feature, target)
print("grid all best params: {}, grid all best score: {}".format(grid_all.best_params_, grid_all.best_score_))
"""{'max_depth': 7, 'min_samples_split': 40}, 0.77025"""

# Use randomized grid seacrh cv for more than one param
clf = DecisionTreeClassifier()
params_dict = dict(max_depth=list(range(1, 10)), min_samples_split=list(range(10, 50, 10)))
rand = RandomizedSearchCV(DecisionTreeClassifier(), params_dict, cv=10, scoring="accuracy", n_iter=10, random_state=5)
rand.fit(feature, target)
print("rand best params: {}, rand best score: {}".format(rand.best_params_, rand.best_score_))
"""{'min_samples_split': 40, 'max_depth': 6}, 0.7663"""

#######
# Final Model Evaluation #
#######
print('FINAL MODEL')

clf_final = DecisionTreeClassifier(max_depth=7, min_samples_split=40)
clf_final.fit(feature_train, target_train)
pred_train_final = clf_final.predict(feature_train)
pred_test_final = clf_final.predict(feature_test)

acc_score_train_final = accuracy_score(target_train, pred_train_final)
acc_score_test_final = accuracy_score(target_test, pred_test_final)
print('acc_score_train {}, acc_score_test {}'.format(acc_score_train_final, acc_score_test_final))
print(confusion_matrix(target_test, pred_test_final))

# Cross validation score don't use the splitted train and test values. it randomly splits train and test 5 times.
cv_score = cross_val_score(clf, feature, target, cv=5)
print('cross validation score {}:'.format(cv_score.mean()))

fpr, tpr, thresholds = roc_curve(target_test, pred_test_final)
auc = metrics.auc(fpr, tpr)
print('auc {}'.format(auc))

sensitivity = tpr[1]
specificity = 1 - fpr[1]
print('sensitivity {}: , specificity {}:'.format(sensitivity, specificity))

# plot roc curve
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
fig = plt.gcf()
name = 'roc_curve_house_final'
save_fig(fig, name, 'Decision_Tree/images')