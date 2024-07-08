from google.colab import files
uploaded = files.upload()

import pandas as pd
import io
dataset = pd.read_csv(io.BytesIO(uploaded['Iris.csv']))

import pandas as pd
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import*
dataset.head()

dataset.info()

dataset.columns

dataset.describe()

dataset.shape

dataset.isnull().sum()

dataset.drop('Id', axis = 1, inplace=True)

dataset.columns


y=dataset['Species']
x=dataset.drop(['Species'],axis=1)


correlation=x.corr()
print(correlation)

plt.figure(figsize=(5,5))
sns.heatmap(correlation,annot=True,cmap="PiYG")
plt.title("Heatmap for correlation")
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.5,random_state=1554)
models={
        "KNN":KNeighborsClassifier(),
        "SVM":SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "LogisticRegresssion": LogisticRegression()
        }
for i in range(len(list(models))):
  model=list(models.values())[i]
  model.fit(x_train,y_train)
  y_train_pred=model.predict(x_train)
  y_test_pred=model.predict(x_test)
  model_train_accuracy=accuracy_score(y_train,y_train_pred)
  model_test_accuracy=accuracy_score(y_test,y_test_pred)

  print("________________________________________________________________")
  print(list(models)[i])
  print("Model performance for train set")
  print("Accuracy:{:.4f}".format(model_train_accuracy))
  print("Model performance for test set")
  print("Accuracy:{:.4f}".format(model_test_accuracy))
  print("________________________________________________________________")


