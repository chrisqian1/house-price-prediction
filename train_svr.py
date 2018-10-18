#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVR,SVR
from sklearn.model_selection import GridSearchCV,cross_val_score
#import matplotlib.pyplot as plt
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


reg=SVR(C=1,gamma=0.5,epsilon=0.2,kernel='linear')
#print(reg)
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
y_m=reg.predict(X_train)
sum_mean=0
sum_m=0  
for j in range(len(y_pred)):
    sum_mean+=(math.log(y_pred[j])-math.log(y_test.values[j]))**2
for j in range(len(y_m)):
    sum_m+=(math.log(y_m[j])-math.log(y_train.values[j]))**2
sum_erro=np.sqrt(sum_mean/len(y_pred))
sum_erro1=np.sqrt(sum_m/len(y_m))
print(sum_erro)
print(sum_erro1)
print(time.time()-a)

