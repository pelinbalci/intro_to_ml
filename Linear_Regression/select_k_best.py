import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# Get pandas data
df_train = pd.read_csv('california_housing_train.csv')
df_test = pd.read_csv('california_housing_test.csv')
df = pd.concat([df_train, df_test], ignore_index=True)
df = df.reindex(np.random.permutation(df.index))

# Select X and y, scale, split
X = df.drop(['median_house_value'], axis=1)
y = df['median_house_value']
std_scaler = MinMaxScaler()
X_scaled = std_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# select best features from train data.
selector = SelectKBest(chi2, k=5)
X_train_new = selector.fit_transform(X_train, y_train)
cols = selector.get_support(indices=True)
cols_name = X.columns[[cols]]
X_test_new = X_test[:,list(cols)]

reg = LinearRegression()
reg.fit(X_train_new, y_train)
pred = reg.predict(X_train_new)
print('train r^2: ', reg.score(X_test_new, y_test))  # 0.64
print('done')
