#!/usr/bin/python

import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from Linear_Regression.model import model_reg
from sklearn.metrics import mean_squared_error, r2_score

from Linear_Regression.age_net_worths import ageNetWorthData
from save_fig import save_fig

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()
reg = model_reg(ages_train, net_worths_train)

plt.figure(figsize=(15,10))
plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")
fig = plt.gcf()
name = 'lin_reg_plot'
save_fig(fig, name, 'Linear_Regression/images/')

pred_test = reg.predict(ages_test)
pred_train = reg.predict(ages_train)

# how to predict?
print('prediction {}'.format(reg.predict(ages_test.reshape(-1,1)[[0]])))
print('prediction random number {}'.format(reg.predict([[30]])))
"""
prediction [[352.19151941]]
prediction random number [[181.83635445]]
"""

# what is the formula?
print('coefficient (w) {}'.format(reg.coef_))
print('intercept (b) {}'.format(reg.intercept_))
print(reg.coef_, ' * x + ', reg.intercept_)
"""
coefficient (w) [[6.30945055]]
intercept (b) [-7.44716216]
[[6.30945055]]  * x +  [-7.44716216]
"""

# r squared score --> the higher is better.
print('train r^2 score:', round(reg.score(ages_train, net_worths_train),3))
print('test r^2 score:', round(reg.score(ages_test, net_worths_test),3))
"""
train r^2 score: 0.877
test r^2 score: 0.789
"""

# The mean squared error & r^2
print('test Mean squared error: %.3f' % mean_squared_error(net_worths_test, pred_test))
print('test r^2, coefficient of determination: %.3f' % r2_score(net_worths_test, pred_test))
"""
test Mean squared error: 1999.097
test r^2, coefficient of determination: 0.789
"""

print('train Mean squared error: %.3f' % mean_squared_error(net_worths_train, pred_train))
print('train r^2, coefficient of determination: %.3f' % r2_score(net_worths_train, pred_train))
"""
train Mean squared error: 1075.460
train r^2, coefficient of determination: 0.877
"""