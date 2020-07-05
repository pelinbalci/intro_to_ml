""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.

    The objective of this exercise is to recreate the decision
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """

from common.prep_terrain_data import makeTerrainData
from common.class_vis import prettyPicture
from save_fig import save_fig


from Naive_Bayes.classify_NB import classify
from Naive_Bayes.calculate_accuracy import NBAccuracy

X, y, features_train, labels_train, features_test, labels_test = makeTerrainData()

# # features are [grade: slope(meyil)] and [bumpy: (tümsek)], two classes: fast or slow.
clf = classify(features_train, labels_train)

# draw and save the decision boundary with the text points overlaid
fig = prettyPicture(clf, features_test, labels_test)
name = "nb_test"
save_fig(fig, name, 'Naive_Bayes/')


# Accuracy
def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy


accuracy = submitAccuracy()
print('accuracy is', accuracy)




