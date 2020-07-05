import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


df_train = pd.read_csv('california_housing_train.csv')
df_test = pd.read_csv('california_housing_test.csv')

df_train_shuff = df_train.reindex(np.random.permutation(df_train.index))

X_train = df_train_shuff.drop(['median_house_value'], axis=1)
#X_train = df_train_shuff[['population', 'total_rooms', 'total_bedrooms', 'median_income']]
y_train = df_train_shuff['median_house_value']

X_valid = df_test.drop(['median_house_value'], axis=1)
#X_valid = df_test[['population', 'total_rooms', 'total_bedrooms', 'median_income']]
y_valid = df_test['median_house_value']

reg = LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_valid)
print('train r^2: ', reg.score(X_train, y_train))  # 0.64
print('valid r^2: ', reg.score(X_valid, y_valid))  # 0.63
lin_reg_rmse = np.sqrt(mean_squared_error(reg.predict(X_valid), y_valid))

ridge = Ridge(alpha=7)
ridge.fit(X_train, y_train)
Ridge_rmse_test = np.sqrt(mean_squared_error(ridge.predict(X_valid), y_valid))

lasso = Lasso(alpha=1)
lasso.fit(X_train, y_train)
Lasso_rmse_test = np.sqrt(mean_squared_error(lasso.predict(X_valid), y_valid))

print('                       OLS    Ridge   Lasso')
print('R^2 value:          ', round(reg.score(X_valid, y_valid), 3), round(ridge.score(X_valid, y_valid), 3),
      round(lasso.score(X_valid, y_valid), 3))
print('RMSE on test  data: ', round(lin_reg_rmse, 3), round(Ridge_rmse_test, 3), round(Lasso_rmse_test, 3))


print('                Predictor      Ridge coeff\'s     Lasso coeff\'s:')
for idx, col_name in enumerate(X_train.columns):
    print('%25s      %13.10f     %13.10f' % (col_name, ridge.coef_[idx], lasso.coef_[idx]))


###### Use grid search cv
alpha_list = [0.0001, 0.001, 0.01, 0.1, 1]

param = {'alpha': alpha_list}

lasso = Lasso()
grid_lasso = GridSearchCV(lasso, param, cv=5)
grid_lasso.fit(X_train, y_train)
print(grid_lasso.best_params_)
print(grid_lasso.best_score_)


ridge = Ridge()
grid_ridge = GridSearchCV(lasso, param, cv=5)
grid_ridge.fit(X_train, y_train)
print(grid_ridge.best_params_)
print(grid_ridge.best_score_)


### Apply the best score
lasso = Lasso(alpha=1)
lasso.fit(X_train, y_train)
print(lasso.score(X_valid, y_valid))

ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
print(ridge.score(X_valid, y_valid))
