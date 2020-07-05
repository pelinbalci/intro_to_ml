
########
# MANUAL WAY
########

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

scale_array = (arr - arr.min()) / (arr.max() - arr.min())
print(scale_array)

#########
# SKLEARN
#########

import numpy as np
from sklearn.preprocessing import MinMaxScaler

weights = np.array([115, 140, 175]).reshape(-1,1)
scaler = MinMaxScaler()
scale_weights = scaler.fit_transform(weights)

print(scale_weights)


#######
# Linear Regression
#######

import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# create data
np.random.seed(42)

X1_np = 100*np.random.rand(125, 1)
X2_np = 5*np.random.rand(125, 1)

X_np = np.concatenate((X1_np, X2_np), axis=1)
y_np = 3*X1_np+4+np.random.rand(125, 1)

# Find lin reg without scaling
X, X_test, y, y_test = train_test_split(X_np, y_np, test_size=0.20)
reg = LinearRegression()
reg.fit(X, y)
print('formula without scaling: ', reg.coef_[0][0], ' * X_1 + ',reg.coef_[0][1], ' * X_2 + ', reg.intercept_)
print('r^2 without scaling: ', reg.score(X_test, y_test))

# Find lin reg with scaling
scaler = MinMaxScaler()
X_np_scaled = scaler.fit_transform(X_np)
y_np_scaled = scaler.fit_transform(y_np)

X, X_test, y, y_test = train_test_split(X_np_scaled, y_np_scaled, test_size=0.20)

reg = LinearRegression()
reg.fit(X, y)
print('formula WITH scaling: ', reg.coef_[0][0], ' * X_1 + ',reg.coef_[0][1], ' * X_2 + ', reg.intercept_)
print('r^2 WIHT scaling: ', reg.score(X_test, y_test))


"""
when we scale X:
formula without scaling:  2.9987053469258518  * X_1 +  -0.02018537388339749  * X_2 +  [4.61044463]
r^2 without scaling:  0.999987158037125
formula WITH scaling:  294.3592941491813  * X_1 +  -0.03313438925336405  * X_2 +  [6.1901725]
r^2 WIHT scaling:  0.9999891053058236

When we scale both X and y:
formula without scaling:  2.9987053469258518  * X_1 +  -0.02018537388339749  * X_2 +  [4.61044463]
r^2 without scaling:  0.999987158037125
formula WITH scaling:  1.0012238211127176  * X_1 +  -0.00011270219924388484  * X_2 +  [-0.00043082]
r^2 WIHT scaling:  0.9999891053058236
"""



#######
# SVM
#######

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

# create data
np.random.seed(42)

X1_np = 100*np.random.rand(125, 1)
X2_np = 5*np.random.rand(125, 1)

X_np = np.concatenate((X1_np, X2_np), axis=1)
y_np = 3*X1_np+4+np.random.rand(125, 1)

# Find svm reg without scaling
X, X_test, y, y_test = train_test_split(X_np, y_np, test_size=0.20)
svm_reg = SVR(kernel = 'rbf')
svm_reg.fit(X, y)
print('svm r^2 without scaling: ', svm_reg.score(X_test, y_test))  # 0.49295716488728447

# Find svm reg with scaling
scaler = MinMaxScaler()
X_np_scaled = scaler.fit_transform(X_np)
y_np_scaled = scaler.fit_transform(y_np)
#y_np_scaled = y_np
X, X_test, y, y_test = train_test_split(X_np_scaled, y_np_scaled, test_size=0.20)
svm_reg = SVR(kernel = 'rbf')
svm_reg.fit(X, y)
print('svm r^2 WITH scaling: ', svm_reg.score(X_test, y_test))  # 0.9560086022599282

# Find svm reg with standart scaling
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_np_std_scaled = std_scaler.fit_transform(X_np)
y_np_std_scaled = std_scaler.fit_transform(y_np)
#y_np_scaled = y_np

X, X_test, y, y_test = train_test_split(X_np_std_scaled, y_np_std_scaled, test_size=0.20)
svm_reg = SVR(kernel = 'rbf')
svm_reg.fit(X, y)
print('svm r^2 WITH std scaling: ', svm_reg.score(X_test, y_test))  # 0.9925541697456666


"""
svm r^2 without scaling:  0.49295716488728447

When we scale only X:
svm r^2 WITH scaling:  0.2667432350042612
svm r^2 WITH std scaling:  0.9925541697456666

When we scale both X and y:
svm r^2 WITH scaling:  0.9560086022599282
svm r^2 WITH std scaling:  0.9925541697456666
"""