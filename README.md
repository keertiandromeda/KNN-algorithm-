# KNN-algorithm-
#The K-Nearest Neighbors (KNN) algorithm is a supervised machine learning method employed to tackle classification and regression problems.
import numpy as np
import pandas as pd
df=pd.read_csv(r'D:\bangalore house price prediction OHE-data.csv')
df
#data split
x=df.drop('price',axis=1)
y=df['price']
print("shape of x",x.shape)
print("shape of y",y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2)
print("Shape of x_train",x_train.shape)
print("Shape of y_train", y_train.shape)
print("Shape of y_train", y_test.shape)
#import knn 
from sklearn.neighbors import KNeighborsRegressor
reg=KNeighborsRegressor(n_neighbors=5)
reg.fit(x_train, y_train)
reg.score(x_test, y_test)


