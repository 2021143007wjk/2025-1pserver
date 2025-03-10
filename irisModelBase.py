#2025.3.10.
#프로젝트 붓꽃분류
#이용희교수님과 열심히 해보자
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

iris_df = pd.read_csv('iris.csv')
print(iris_df)
print(iris_df['sepal_length'])

y= iris_df['species']
print(y)

x= iris_df.drop('species', axis=1)
print(x)

kn = KNeighborsClassifier()
model_kn = kn.fit(x, y)

# x_new = np.array([[3,3,3,3]])
#
x_new = np.array([[5.0,3.4,1.4,0.2]])
# setosa [[1. 0. 0.]]
pred = model_kn.predict(x_new)
print(pred)
probavility = model_kn.predict_proba(x_new)
print(probavility)