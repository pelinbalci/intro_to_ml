import sys
from common.class_vis import prettyPicture
from common.prep_terrain_data import makeTerrainData
from save_fig import save_fig

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# create data
features_train, labels_train, features_test, labels_test = makeTerrainData()


def decisiontree_trial(features_train, labels_train, features_test, clf):
    clf.fit(features_train, labels_train)
    pred_train = clf.predict(features_train)
    pred_test = clf.predict(features_test)
    return clf, pred_train, pred_test


def evaluate_results(pred_test, pred_train, labels_test, labels_train, name):
    accuracy_test = accuracy_score(labels_test, pred_test)
    accuracy_train = accuracy_score(labels_train, pred_train)
    fig = prettyPicture(clf, features_test, labels_test)
    save_fig(fig, name, 'Decision_Tree/images')
    print(name, 'test accuracy {}:, train accuracy {}:'.format(round(accuracy_test, 3), round(accuracy_train, 3)))
    print('confusion_matrix {}:'.format(confusion_matrix(labels_test, pred_test)))


clf = DecisionTreeClassifier(max_depth=2)
name = 'max_depth 2'
clf, pred_train, pred_test = decisiontree_trial(features_train, labels_train, features_test, clf)
evaluate_results(pred_test, pred_train, labels_test, labels_train, name)

clf = DecisionTreeClassifier(max_depth=10)
name = 'max_depth 10'
clf, pred_train, pred_test = decisiontree_trial(features_train, labels_train, features_test, clf)
evaluate_results(pred_test, pred_train, labels_test, labels_train, name)

clf = DecisionTreeClassifier(max_depth=100)
name = 'max_depth 100'
clf, pred_train, pred_test = decisiontree_trial(features_train, labels_train, features_test, clf)
evaluate_results(pred_test, pred_train, labels_test, labels_train, name)

clf = DecisionTreeClassifier(min_samples_split=2)
name = 'min_sample_split 2'
clf, pred_train, pred_test = decisiontree_trial(features_train, labels_train, features_test, clf)
evaluate_results(pred_test, pred_train, labels_test, labels_train, name)

clf = DecisionTreeClassifier(min_samples_split=50)
name = 'min_sample_split 50'
clf, pred_train, pred_test = decisiontree_trial(features_train, labels_train, features_test, clf)
evaluate_results(pred_test, pred_train, labels_test, labels_train, name)

clf = DecisionTreeClassifier(min_samples_split=100)
name = 'min_sample_split 100'
clf, pred_train, pred_test = decisiontree_trial(features_train, labels_train, features_test, clf)
evaluate_results(pred_test, pred_train, labels_test, labels_train, name)

clf = DecisionTreeClassifier(min_samples_split=20, max_depth=10)
name = 'min_sample_split 10, max_depth 10'
clf, pred_train, pred_test = decisiontree_trial(features_train, labels_train, features_test, clf)
evaluate_results(pred_test, pred_train, labels_test, labels_train, name)

"""
accuracy for model max_depth 2 is 0.848  --> may be underfits
accuracy for model max_depth 10 is 0.912  ----> overfits
accuracy for model min_sample_split 2 is 0.908   --> overfits to train data. test accuracy is bad. 
accuracy for model min_sample_split 50 is 0.912 --> ok.
accuracy for model min_sample_split 10, max_depth 10 is 0.924 --> best
"""