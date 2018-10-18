#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
import time

a=time.time()
train_1=pd.read_csv('train_0.csv')
test_1=pd.read_csv('test_1.csv')
del train_1[list(train_1)[0]]
del test_1[list(test_1)[0]]
#print(test_1)

X=train_1.ix[0:train_1.shape[0],1:train_1.shape[1]-1]
y=train_1['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
#print(X)
X1=test_1.ix[0:test_1.shape[0],1:test_1.shape[1]]
X2=X1.reindex(columns=list(X)).fillna(0)

reg=ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],n_alphas=100,alphas=np.linspace(1,1000,1000),cv=3)
reg.fit(X,y)
y_pred=reg.predict(X_test)
y_m=reg.predict(X_train)
sum_mean=0
sum_m=0
for i in range(len(y_pred)):
    sum_mean+=(math.log(y_pred[i])-math.log(y_test.values[i]))**2
for i in range(len(y_m)):
    sum_m+=(math.log(y_m[i])-math.log(y_train.values[i]))**2
sum_erro=np.sqrt(sum_mean/365)
sum_erro1=np.sqrt(sum_m/len(y_m))
print("RMSE by hand:"+str(sum_erro))
print("RMSE:"+str(sum_erro1))
y_p=reg.predict(X2)
pred=pd.DataFrame(index=np.arange(test_1.shape[0]))
pred.insert(0,'Id',test_1['Id'])
pred.insert(1,'SalePrice',y_p)
pred.to_csv('Price-enet.csv')
print(time.time()-a)

