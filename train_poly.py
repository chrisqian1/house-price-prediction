#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV


train_1=pd.read_csv('train_2.csv')
train_2=pd.read_csv('train_3.csv')
del train_1[list(train_1)[0]]
del train_2[list(train_2)[0]]
test_1=pd.read_csv('test_2.csv')
test_2=pd.read_csv('test_3.csv')
del test_1[list(test_1)[0]]
del test_2[list(test_2)[0]]
Id=test_2['Id']
print(train_1)

X1=train_1.ix[0:train_1.shape[0],0:train_1.shape[1]]
X2=train_2.ix[0:train_2.shape[0],0:train_2.shape[1]-1]

y=train_2['SalePrice']
0.8
#print(X2.ix[0:X2.shape[0],1]*X2.ix[0:X2.shape[0],2])

X2_shape=X2.shape[1]

for i in range(1,X2_shape):
    for j in range(i,X2_shape):
        X2.insert(X2.shape[1],list(X2)[i]+'_'+list(X2)[j],X2.ix[0:X2.shape[0],i]*X2.ix[0:X2.shape[0],j]) 
X_train=X1.merge(X2,on='Id')
print(X_train)
#print(X_train)
#X2.to_csv('train_poly.csv')

test_shape=test_2.shape[1]
for i in range(1,test_shape):
    for j in range(i,test_shape):
        test_2.insert(test_2.shape[1],list(test_2)[i]+'_'+list(test_2)[j],test_2.ix[0:test_2.shape[0],i]*test_2.ix[0:test_2.shape[0],j])
X3=test_1.merge(test_2,on='Id')
X4=X3.reindex(columns=list(X_train)).fillna(0)
print(X4)
#X_train,X_test,y_train,y_test=train_test_split(X2,y,random_state=1)

#reg=LinearRegression()
reg=LassoCV(alphas=np.linspace(300,500,20))
reg.fit(X_train,y)

#y_pred=reg.predict(X_test)
#sum_mean=0
#for i in range(len(y_pred)):
    #sum_mean+=(math.log(y_pred[i])-math.log(y_test.values[i]))**2
#sum_erro=np.sqrt(sum_mean/365)
#print("RMSE by hand:"+str(sum_erro))

y_p=reg.predict(X4)
pred=pd.DataFrame(index=np.arange(test_1.shape[0]))
pred.insert(0,'Id',test_1['Id'])
pred.insert(1,'SalePrice',y_p)

pred.to_csv('Price-poly.csv')

#test_1=pd.read_csv('test_2.csv')
#test_2=pd.read_csv('test_3.csv')
#Id_1=test_2.pop('Id')
#del test_1[list(test_1)[0]]
#del test_2[list(test_2)[0]]

#X_1=test_1.ix[0:train_1.shape[0],0:train_1.shape[1]]
#X_2=test_2.ix[0:train_1.shape[0],0:train_1.shape[1]]

#print(X1)
#print(X_1)
#print(list(X2))
#print(list(X_2))

