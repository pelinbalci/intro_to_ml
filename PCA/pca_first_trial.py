import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('california_housing_train.csv')
df_test = pd.read_csv('california_housing_test.csv')
df = pd.concat([df_train, df_test], ignore_index=True)
df = df.reindex(np.random.permutation(df.index))
print(df.columns)
# Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
# 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value'], dtype='object')

# Select X and y, split
X = df.drop(['median_house_value'], axis=1)
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Lin Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print('train r^2: ', reg.score(X_train, y_train))  # 0.64
print('test r^2: ', reg.score(X_test, y_test))  # 0.63
lin_reg_rmse = np.sqrt(mean_squared_error(reg.predict(X_test), y_test))
print('test rmse', lin_reg_rmse)  # 69982

# PCA
pca = PCA(n_components=3)
pca.fit(X_train)
print('eigen values:', pca.explained_variance_ratio_)
first_pca = pca.components_[0]
second_pca = pca.components_[1]
third_pca = pca.components_[2]

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Lin Regression
reg = LinearRegression()
reg.fit(X_train_pca, y_train)
pred = reg.predict(X_test_pca)
print('pca train r^2: ', reg.score(X_train_pca, y_train))  # 0.64
print('pca test r^2: ', reg.score(X_test_pca, y_test))  # 0.63
lin_reg_rmse = np.sqrt(mean_squared_error(reg.predict(X_test_pca), y_test))
print('pca test rmse', lin_reg_rmse)  # 69982


# PCA
pca = PCA(0.95)
pca.fit(X_train)
print('eigen values:', pca.explained_variance_ratio_)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Lin Regression
reg = LinearRegression()
reg.fit(X_train_pca, y_train)
pred = reg.predict(X_test_pca)
print('pca train r^2: ', reg.score(X_train_pca, y_train))  # 0.64
print('pca test r^2: ', reg.score(X_test_pca, y_test))  # 0.63
lin_reg_rmse = np.sqrt(mean_squared_error(reg.predict(X_test_pca), y_test))
print('pca test rmse', lin_reg_rmse)  # 69982