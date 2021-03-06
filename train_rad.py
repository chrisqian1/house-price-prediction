#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time

a=time.time()
train_1=pd.read_csv('train_0.csv')
test_1=pd.read_csv('test_1.csv')
del train_1[list(train_1)[0]]
del test_1[list(test_1)[0]]
X=train_1.ix[0:train_1.shape[0],1:train_1.shape[1]-1]
y=train_1['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
#print(X)
X1=test_1.ix[0:test_1.shape[0],1:test_1.shape[1]]
X2=X1.reindex(columns=list(X)).fillna(0)
sum_erro=np.zeros((10,))
sum_erro1=np.zeros((10,))
for i in range(0,10,1):
    reg=RandomForestRegressor(n_estimators=100,oob_score=True,max_features=0.7,max_depth=30,max_leaf_nodes=150,min_samples_split=3)
    reg.fit(X_train,y_train)
    y_pred=reg.predict(X_test)
    y_m=reg.predict(X_train)
#reg.fit(X,y)
#y_p=reg.predict(X2)
#pred=pd.DataFrame(index=np.arange(test_1.shape[0]))
#pred.insert(0,'Id',test_1['Id'])
#pred.insert(1,'SalePrice',y_p)
#pred.to_csv('Price-ada.csv')
    sum_mean=0
    sum_m=0  

    for j in range(len(y_pred)):
        sum_mean+=(math.log(y_pred[j])-math.log(y_test.values[j]))**2
    for j in range(len(y_m)):
        sum_m+=(math.log(y_m[j])-math.log(y_train.values[j]))**2
    sum_erro[i]=np.sqrt(sum_mean/len(y_pred))
    sum_erro1[i]=np.sqrt(sum_m/len(y_m))
    print(sum_erro[i])
print(sum_erro)
print(sum_erro1)
reg.fit(X,y)
y_p=reg.predict(X2)
pred=pd.DataFrame(index=np.arange(test_1.shape[0]))
pred.insert(0,'Id',test_1['Id'])
pred.insert(1,'SalePrice',y_p)
pred.to_csv('Price-rad.csv')

print(time.time()-a)
