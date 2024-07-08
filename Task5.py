from google.colab import files
uploaded = files.upload()

import pandas as pd
import io
df = pd.read_csv(io.BytesIO(uploaded['Advertising.csv']))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor

df.info()

df.head()

df.describe()

df.columns

df.drop('Unnamed: 0',inplace=True,axis=1)

df.head()


df.columns

print(df.isnull().sum())

sns.pairplot(data=df)
correlation=df.corr()
sns.heatmap(correlation, annot=True)
plt.show()

y=df['Sales'];
x=df[['TV', 'Radio', 'Newspaper']]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=40)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')

model=KNeighborsRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')