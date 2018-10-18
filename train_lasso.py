#usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
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
#print(X1)
#print(X2)
al=80
lasso=Lasso(alpha=al)
#lasso=LassoCV(alphas=np.linspace(1,1000,1000))


lasso.fit(X,y)
#print(lasso.alpha_)
lasso_1=Lasso(alpha=al)
lasso_1.fit(X,y)
y_p=lasso.predict(X2)

corr=X.corrwith(y,0)
cor1=pd.DataFrame(index=range(1,11),columns=['max','maxv','abs','absv'])
corr1=corr
for i in range(X.shape[1]):
    cor1.ix[i,0]=corr1.idxmax()
    cor1.ix[i,1]=corr1.max()
    corr1=corr1.drop(corr1.idxmax())
corr2=corr
corr3=corr.abs()
for i in range(X.shape[1]):
    cor1.ix[i,2]=corr3.idxmin()
    cor1.ix[i,3]=corr3.min()
    corr3=corr3.drop(corr3.idxmin())
cor1.to_csv('corr.csv')
print(lasso.coef_)
#print(coef)

mask=lasso.coef_!=0
l=list(X)
for i in range(X.shape[1]):
    if mask[i]==False:
        del X[l[i]]
#print(lasso.alpha_)
X3=X2.reindex(columns=list(X))
X.insert(0,'Id',train_1['Id'])
X.insert(X.shape[1],'SalePrice',y)
X3.insert(0,'Id',test_1['Id'])

y_pred=lasso.predict(X_test)
#y_m=lasso.predict(X_train)
#pred=pd.DataFrame(index=np.arange(test_1.shape[0])):
#pred.insert(0,'Id',test_1['Id'])
#pred.insert(1,'SalePrice',y_p)

#pred.to_csv('Price-lasso.csv')

print(metrics.r2_score(y_test,y_pred))


#m=y_pred.mean()

#sum_mean=0
#sum_m=0
#for i in range(len(y_pred)):
    #sum_mean+=(math.log(y_pred[i])-math.log(y_test.values[i]))**2
#for i in range(len(y_m)):
    #sum_m+=(math.log(y_m[i])-math.log(y_train.values[i]))**2
#sum_erro=np.sqrt(sum_mean/365)
#sum_erro1=np.sqrt(sum_m/len(y_m))
#print("RMSE by hand:"+str(sum_erro))
#print("RMSE:"+str(sum_erro1))
#print(m)
X.to_csv('train_lasso.csv')
X3.to_csv('test_lasso.csv')
print(time.time()-a)
