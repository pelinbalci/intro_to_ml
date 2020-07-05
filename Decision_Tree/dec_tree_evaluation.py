import sys
from common.class_vis import prettyPicture
from common.prep_terrain_data import makeTerrainData
from save_fig import save_fig

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
# create data
features_train, labels_train, features_test, labels_test = makeTerrainData()

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

"""
max_depth 2 test accuracy 0.848:, train accuracy 0.857:

         pred 0   pred 1

actual 0 [[ 79       5]
actual  1 [ 33      133]]

         pred 0  pred 1

actual 0 [[ TN      FP]
actual  1 [ FN      TP]]

accuracy = TN + TP / all = 212/250 = 0.848

sensitivity = recall = TPR = TP/ TP + FN = 133/ 166 = 0.8012
specificity = TN / TN + FP  = 79/84 = 0.94 
FPR = 1- specificity = 1-0.94 = 0.0595

precision = TP / TP + FP = 133/138 = 0.9637

labels_test class 1 = 166
labels_test class 0 = 84

pred_test class 1 = 138
pred_test class 0 = 112


              precision    recall  f1-score   support

           0       0.71      0.94      0.81        84
           1       0.96      0.80      0.88       166

    accuracy                           0.85       250
   macro avg       0.83      0.87      0.84       250
weighted avg       0.88      0.85      0.85       250
"""