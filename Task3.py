from google.colab import files
uploaded = files.upload()

import pandas as pd
import io
dataset = pd.read_csv(io.BytesIO(uploaded['car data.csv']))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


dataset.head()


dataset.info()


dataset.describe()


dataset.isnull().sum()


dataset.duplicated().sum()


dataset.drop_duplicates(inplace=True)


dataset.columns

dataset["Fuel_Type"].unique()


dataset["Selling_type"].unique()

dataset["Fuel_Type"].replace({'Petrol':0, 'Diesel':1, 'CNG':2},inplace=True)
dataset["Selling_type"].replace({'Dealer':0,'Individual':1},inplace=True)
dataset["Transmission"].replace({'Manual':0, 'Automatic':1},inplace=True)


dataset["Fuel_Type"].unique()

dataset["Transmission"].unique()

dataset.describe()

dataset.head()

t=dataset.drop(['Car_Name'],axis=1)

correlation=t.corr()
print(correlation)

plt.figure(figsize=(15,8))
sns.heatmap(correlation,annot=True,cmap="PiYG")
plt.title("Heatmap for correlation")
plt.show()

y=dataset['Selling_Price']
x=dataset.drop(['Car_Name','Selling_Price','Selling_type'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=50)


model=RandomForestRegressor(n_estimators=100, random_state=50)

predict=model.fit(x_train,y_train)
y_pred=model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')

model=KNeighborsRegressor()
predict=model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')