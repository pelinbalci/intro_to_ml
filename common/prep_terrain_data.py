#!/usr/bin/python
import random

def makeTerrainData(n_points=1000):
    """make the toy dataset """
    random.seed(42)
    grade = [random.random() for i in range(0, n_points)]  #[0.63, 0.025, 0.275, 0.223, 0.736, 0.676, 0.89, 0.085, 0.42, 0.029]
    bumpy = [random.random() for i in range(0, n_points)]  #[0.218, 0.50, 0.026, 0.19, 0.649, 0.54, 0.22, 0.58, 0.809, 0.006]
    error = [random.random() for i in range(0, n_points)]
    y = [round(grade[i]*bumpy[i]+0.3+0.1*error[i]) for i in range(0, n_points)] #[1, 0, 0, 0, 1, 1, 1, 0, 1, 0]
    for i in range(0, len(y)):
        if grade[i] > 0.8 or bumpy[i] > 0.8:
            y[i] = 1.0  # <class 'list'>: [1, 0, 0, 0, 1, 1, 1.0, 0, 1.0, 0]

    # split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75 * n_points)
    X_train = X[0:split]  # [[0.63, 0.218], [0.025, 0.50] ... ]
    X_test = X[split:]
    y_train = y[0:split]  # [1, 0, 0, 0, 1, 1, 1.0]
    y_test = y[split:]

    grade_sig = [X_train[i][0] for i in range(0, len(X_train)) if y_train[i] == 0]
    bumpy_sig = [X_train[i][1] for i in range(0, len(X_train)) if y_train[i] == 0]
    grade_bkg = [X_train[i][0] for i in range(0, len(X_train)) if y_train[i] == 1]
    bumpy_bkg = [X_train[i][1] for i in range(0, len(X_train)) if y_train[i] == 1]

    grade_sig = [X_test[i][0] for i in range(0, len(X_test)) if y_test[i] == 0]
    bumpy_sig = [X_test[i][1] for i in range(0, len(X_test)) if y_test[i] == 0]
    grade_bkg = [X_test[i][0] for i in range(0, len(X_test)) if y_test[i] == 1]
    bumpy_bkg = [X_test[i][1] for i in range(0, len(X_test)) if y_test[i] == 1]

    test_data = {"fast": {"grade": grade_sig, "bumpiness": bumpy_sig},
                 "slow": {"grade": grade_bkg, "bumpiness": bumpy_bkg}}

    return X, y, X_train, y_train, X_test, y_test