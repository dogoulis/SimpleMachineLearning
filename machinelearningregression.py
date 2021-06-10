# Importing the libraries
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

# Reading the dataset

data = pd.read_csv('fuel_emissions.csv')

# Data Preprocessing

# finding the missing values


print(data.info())
print(data.isnull().sum())

#encoding cat. variables

data = data.drop(['tax_band', 'file', 'manufacturer', 'model', 'description', 'transmission', 'transmission_type', 'fuel_type'],axis=1)
df1 = data.pop('fuel_cost_12000_miles')
data['fuel_cost_12000_miles'] = df1
print(data.info())

# imputing
KNN = KNNImputer(n_neighbors=3)
data = KNN.fit_transform(data)
data = pd.DataFrame(data)

X = data.iloc[:,:-1].values
X = pd.DataFrame(X)
y = data.iloc[:,-1].values

# Splitting the dataset and preprocessing

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state = 42)

# scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# pca 
pca = PCA(n_components=0.9)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(np.shape(x_train))
print(np.shape(y_train))
print(np.any(np.isnan(x_train)))
print(np.any(np.isnan(y_train)))

# Model Implementation

# linear regression


reg = LinearRegression()
reg.fit(x_train, y_train)
y_reg_pred = reg.predict(x_test)

# randomForestRegressor

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_rf_pred = rf.predict(x_test)

# neighbors regression

kn = KNeighborsRegressor(n_neighbors=7, weights='distance')
kn.fit(x_train, y_train)
y_kn_pred = kn.predict(x_test)

# Model Evaluation

print('MAE for linear reg is: ', mean_absolute_error(y_test, y_reg_pred))
print('MSE for linear reg is: ', mean_squared_error(y_test, y_reg_pred))
print('Explained Variance for linear reg is: ', explained_variance_score(y_test, y_reg_pred))

print('MAE for random forest is: ', mean_absolute_error(y_test, y_rf_pred))
print('MSE for random forest is: ', mean_squared_error(y_test, y_rf_pred))
print('Explained Variance for random forest is: ', explained_variance_score(y_test, y_rf_pred))

print('MAE for neighbors is: ', mean_absolute_error(y_test, y_kn_pred))
print('MSE for neighbors is: ', mean_squared_error(y_test, y_kn_pred))
print('Explained Variance for neighbors is: ', explained_variance_score(y_test, y_kn_pred))