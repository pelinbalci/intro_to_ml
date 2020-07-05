from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """

    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    # calculate and return the accuracy on the test data - manual way
    sum_false_pred = 0
    for i in range(len(labels_test)):
        sum_false_pred += abs(pred[i] - labels_test[i])

    total_pred = len(pred)
    accuracy_manual = (total_pred - sum_false_pred) / total_pred

    # calculate and return the accuracy on the test data - sklearn
    accuracy_sklearn = accuracy_score(pred, labels_test)

    # calculate and return the accuracy on the test data - sklearn othr way
    accuracy_sklearn_2 = clf.score(features_test, labels_test)

    print('accuracy_manual {}, sklearn accuracy {}, sklearn score {}'.format(accuracy_manual, accuracy_sklearn, accuracy_sklearn_2))
    return accuracy_sklearn