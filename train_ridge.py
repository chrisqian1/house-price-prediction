#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import RidgeCV
import time
a=time.time()
train_1=pd.read_csv('train_0.csv')
del train_1['Id']
del train_1[list(train_1)[0]]
test_1=pd.read_csv('test_1.csv')
del test_1[list(test_1)[0]]

X=train_1.ix[0:train_1.shape[0],0:train_1.shape[1]-1]
y=train_1['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
X1=test_1.ix[0:test_1.shape[0],1:test_1.shape[1]]
X2=X1.reindex(columns=list(X_train)).fillna(0)

reg=RidgeCV(alphas=np.linspace(0.1,100,1000))
reg.fit(X,y)

y_pred=reg.predict(X_test)
y_m=reg.predict(X_train)
y_p=reg.predict(X2)
sum_mean=0
sum_m=0
for i in range(len(y_pred)):
    #sum_mean+=(y_pred[i]-y_test.values[i])**2
    sum_mean+=(math.log(y_pred[i])-math.log(y_test.values[i]))**2
for i in range(len(y_m)):
    sum_m+=(math.log(y_m[i])-math.log(y_train.values[i]))**2
sum_erro=np.sqrt(sum_mean/365)
sum_erro1=np.sqrt(sum_m/len(y_m))
print("RMSE by hand:"+str(sum_erro))
print("RMSE:"+str(sum_erro1))
print(reg.alpha_)
pred=pd.DataFrame(index=np.arange(test_1.shape[0]))
pred.insert(0,'Id',test_1['Id'])
pred.insert(1,'SalePrice',y_p)

pred.to_csv('Price-ridge.csv')




print(time.time()-a)
