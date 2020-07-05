import numpy as np
import time
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from Linear_Regression.age_net_worths import ageNetWorthData
from sklearn.model_selection import train_test_split

# create data
random.seed(42)
np.random.seed(42)

X_np = 2*np.random.rand(125, 1)
y_np = 3*X_np+4+np.random.rand(125, 1)

X, X_test, y, y_test = train_test_split(X_np, y_np, test_size=0.20)


# Find lin reg with sklearn
reg = LinearRegression()
reg.fit(X, y)
pred = reg.predict(X)
print('formula: ', reg.coef_, ' * X + ', reg.intercept_)
print('r^2: ', reg.score(X, y))


# Closed form solution
onesX = np.ones(100).reshape(-1,1)
X_b = np.concatenate((onesX, X), axis=1)
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('w and b : ', theta)
pred_manual = X_b.dot(theta)


plt.figure()
plt.plot(X, y, "bo")
plt.plot(X, pred, "r-")
plt.plot(X, pred_manual, "g-")
#plt.show()


# Stochastic gradient descent
start = time.time()
onesX = np.ones(100).reshape(-1,1)
X_b = np.concatenate((onesX, X), axis=1)
onesX_test = np.ones(25).reshape(-1,1)
X_b_test = np.concatenate((onesX_test, X_test), axis=1)
#theta = np.random.rand(2, 1)
theta = [[0.31141331], [0.97951053]]
print('initial theta', theta)
#total_obs = len(X_b)
total_obs = 10
eta = 0.1
epochs = 10

plt.figure(figsize=(10,10))
plt.plot(X, y, "bo")
for epoch in range(epochs):
    for i in range(total_obs):
        idx = np.random.randint(0, total_obs)
        x_b_stochastic = X_b[idx:idx + 1]
        y_stochastic = y[idx:idx + 1]
        gradient = (2 / total_obs) * x_b_stochastic.T.dot(x_b_stochastic.dot(theta) - y_stochastic)
        theta = theta - eta * gradient
        print('iter {} ... theta 0: {} theta 1: {}, gradient: {}  {}'.format(i, round(theta[0][0],3),round(theta[1][0],3), round(gradient[0][0],2),
                                                        round(gradient[1][0],2)))

    y_pred = X_b.dot(theta)
    r_square = 1 - (sum((y-y_pred)**2) / sum((y-y.mean())**2))
    mse = 1/total_obs * sum((y-y_pred)**2)
    y_pred_test = X_b_test.dot(theta)
    mse_test = 1 / total_obs * sum((y_test - y_pred_test) ** 2)
    r_square_test = 1 - (sum((y_test - y_pred_test) ** 2) / sum((y_test - y_test.mean()) ** 2))

    print('epoch {} ... theta 0: {} theta 1: {}, gradient: {}  {}, mse: {}, '
          'r^2: {}, mse_test: {}, r^2_test: {}'.format(epoch, round(theta[0][0], 3), round(theta[1][0], 3),
                                                       round(gradient[0][0], 2),
                                                       round(gradient[1][0], 2), round(mse[0], 3),
                                                       round(r_square[0], 3),
                                                       round(mse_test[0], 3), round(r_square_test[0], 3)))
    if epoch == 0:
        plt.plot(X, y_pred, "g-")
    else:
        plt.plot(X, y_pred, "r-")

y_pred = X_b.dot(theta)
plt.plot(X, y_pred, "b-")
plt.show()

print('Duration of stochastic gradient descent {}'.format(time.time() - start))


start = time.time()
# batch gradient descent --> use the whole data
onesX = np.ones(100).reshape(-1,1)
X_b = np.concatenate((onesX, X), axis=1)
onesX_test = np.ones(25).reshape(-1,1)
X_b_test = np.concatenate((onesX_test, X_test), axis=1)
#theta = np.random.rand(2, 1)
theta = [[0.31141331], [0.97951053]]
print('initial theta', theta)
n_iter = 10
eta = 0.1
m = len(X_b)

plt.figure(figsize=(10,10))
plt.plot(X, y, "bo")
for i in range(n_iter):
    gradient = (2 / m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradient
    y_pred = X_b.dot(theta)
    mse = 1/m * sum((y-y_pred)**2)
    r_square = 1 - (sum((y-y_pred)**2) / sum((y-y.mean())**2))
    y_pred_test = X_b_test.dot(theta)
    mse_test = 1 / m * sum((y_test - y_pred_test) ** 2)
    r_square_test = 1 - (sum((y_test - y_pred_test) ** 2) / sum((y_test - y_test.mean()) ** 2))
    if i == 0:
        plt.plot(X, y_pred, "g-")
    elif i <= 10:
        plt.plot(X, y_pred, "r-")

    print('iter {} ... theta 0: {} theta 1: {}, gradient: {}  {}, mse: {}, '
          'r^2: {}, mse_test: {}, r^2_test: {}'.format(i, round(theta[0][0],3),round(theta[1][0],3), round(gradient[0][0],2),
                                                       round(gradient[1][0],2), round(mse[0],3), round(r_square[0],3),
                                                       round(mse_test[0],3), round(r_square_test[0],3)))

y_pred = X_b.dot(theta)
plt.plot(X, y_pred, "b-")
plt.show()
print('Duration of batch gradient descent {}'.format(time.time() - start))

"""
iter 0 ... theta 0: 1.891 theta 1: 2.12, gradient: -12.56  -17.85, mse: 15.591, r^2: -0.119, mse_test: 4.228, r^2_test: -0.073
iter 1 ... theta 0: 2.509 theta 1: 3.084, gradient: -6.18  -9.64, mse: 5.58, r^2: 0.6, mse_test: 1.573, r^2_test: 0.601
iter 2 ... theta 0: 2.794 theta 1: 3.618, gradient: -2.85  -5.34, mse: 2.766, r^2: 0.802, mse_test: 0.805, r^2_test: 0.796

iter 148 ... theta 0: 0.623 theta 1: 6.147, gradient: 0.01  -0.01, mse: 0.08, r^2: 0.994, mse_test: 0.024, r^2_test: 0.994
iter 149 ... theta 0: 0.622 theta 1: 6.148, gradient: 0.01  -0.01, mse: 0.08, r^2: 0.994, mse_test: 0.024, r^2_test: 0.994
"""

