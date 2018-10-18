#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import time
a=time.time()
train_1=pd.read_csv('train_0.csv')
test_1=pd.read_csv('test_1.csv')
del train_1['Id']
del train_1[list(train_1)[0]]
del test_1[list(test_1)[0]]

X=train_1.ix[0:train_1.shape[0],0:train_1.shape[1]-1]
y=train_1['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

#print(X)
#print(y)
X1=test_1.ix[0:test_1.shape[0],1:test_1.shape[1]]
X2=X1.reindex(columns=list(X)).fillna(0)
#param_test1={'n_estimators':range(100,601,20)}
#gsearch=GridSearchCV(estimator=AdaBoostRegressor(base_estimator=DecisionTreeRegressor()),param_grid=param_test1,scoring='roc_auc',cv=3)
#gsearch.fit(X,y)
#print(gsearch.best_params_)

#,'max_features':np.arange(0.2,0.8,0.1),'max_depth':range(10,31,5)
sum_erro=np.zeros((10,))
sum_erro1=np.zeros((10,))
m=1
for i in range(0,10):
    ada=AdaBoostRegressor(DecisionTreeRegressor(max_features=0.8,max_depth=30,max_leaf_nodes=150),n_estimators=500,learning_rate=1,loss='linear')
    ada.fit(X_train,y_train)
    y_pred=ada.predict(X_test)
    y_m=ada.predict(X_train)
#ada.fit(X,y)
#y_p=ada.predict(X2)

#plt.figure()
#plt.plot(np.arange(1,501),err,label='test err',color='red')
#scores=cross_val_score(reg,X_train,y_train)
#print(scores)
#y_pred=reg.predict(X_test)
#y_m=reg.predict(X_test_1)

#y_p=reg.predict(X2)

#pred=pd.DataFrame(index=np.arange(test_1.shape[0]))
#pred.insert(0,'Id',test_1['Id'])
#pred.insert(1,'SalePrice',y_p)
#pred.to_csv('Price-ada1.csv')







    sum_mean=0
    sum_m=0  

    for j in range(len(y_pred)):
        sum_mean+=(math.log(y_pred[j])-math.log(y_test.values[j]))**2
    for j in range(len(y_m)):
        sum_m+=(math.log(y_m[j])-math.log(y_train.values[j]))**2
    sum_erro[i]=np.sqrt(sum_mean/len(y_pred))
    sum_erro1[i]=np.sqrt(sum_m/len(y_m))
    print(sum_erro[i])
    if sum_erro[i]<m:
        ada.fit(X,y)
        y_p=ada.predict(X2)
        m=sum_erro[i]
print(np.mean(sum_erro))
print(np.mean(sum_erro1))
pred=pd.DataFrame(index=np.arange(test_1.shape[0]))
pred.insert(0,'Id',test_1['Id'])
pred.insert(1,'SalePrice',y_p)
pred.to_csv('Price-ada.csv')
print(time.time()-a)

