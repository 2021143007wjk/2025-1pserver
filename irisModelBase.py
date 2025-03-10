#2025.3.10.
#프로젝트 붓꽃분류
#이용희교수님과 열심히 해보자
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

iris_df = pd.read_csv('iris.csv')
print(iris_df)
print(iris_df['sepal_length'])

y= iris_df['species']
print(y)

X= iris_df.drop('species', axis=1)
print(X)

kn = KNeighborsClassifier()
rfc = RandomForestClassifier()

model_kn = kn.fit(X, y)
model_rfc = rfc.fit(X, y)

joblib.dump(model_rfc, 'model_rfc.pkl')

# x_new = np.array([[3,3,3,3]])
# ['versicolor']
# [[0.  0.8 0.2]]
# ['virginica']
# [[0.03 0.42 0.55]]

# x_new = np.array([[5.0,3.4,1.4,0.2]])
# ['setosa']
# [[1. 0. 0.]]
# ['setosa']
# [[1. 0. 0.]]

x_new = np.array([[1,4.2,1.4,7]])
# ['versicolor']
# [[0.2 0.6 0.2]]
# ['setosa']
# [[0.45 0.23 0.32]]

model_rfc = joblib.load('model_rfc.pkl')

pred = model_kn.predict(x_new)
print(pred)
probavility = model_kn.predict_proba(x_new)
print(probavility)
pred = model_rfc.predict(x_new)
print(pred)
probavility = model_rfc.predict_proba(x_new)
print(probavility)