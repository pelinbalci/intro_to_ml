predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

actual_positive_class = sum(true_labels)
actual_negative_class = len(true_labels) - sum(true_labels)

pred_positive_class = sum(predictions)
pred_negative_class = len(predictions) - sum(predictions)

tp, fn, tn, fp = 0, 0, 0, 0
for i in range(len(true_labels)):
    label = true_labels[i]
    pred = predictions[i]

    if (label == 1) and (label == pred):
        tp += 1

    if (label == 1) and (label != pred):
        fn += 1

    if (label == 0) and (label == pred):
        tn += 1

    if (label == 0) and (label != pred):
        fp += 1


recall = tp / actual_positive_class
precision = tp / pred_positive_class

print('recall {}, precision {}'.format(round(recall, 3), round(precision, 3)))