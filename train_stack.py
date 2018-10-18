#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso,LassoCV,LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
import time
torch.manual_seed(1)

a=time.time()
train_1=pd.read_csv('train_0.csv')
test_1=pd.read_csv('test_1.csv')
del train_1[list(train_1)[0]]
del test_1[list(test_1)[0]]

X=train_1.ix[0:train_1.shape[0],1:train_1.shape[1]-1]
y=train_1['SalePrice']

kf=KFold(n_splits=5,random_state=1)

X1=test_1.ix[0:test_1.shape[0],1:test_1.shape[1]]
X2=X1.reindex(columns=list(X)).fillna(0)


al=80
X_t_lasso=pd.Series()
y_t=pd.Series()
X_p_lasso=pd.DataFrame(index=np.arange(X1.shape[0]))
for train,test in kf.split(X):
    X_train,X_test=X.ix[train,:],X.ix[test,:]
    y_train,y_test=y[train],y[test]
    lasso=Lasso(alpha=al)
    lasso.fit(X_train,y_train)
    X_t=pd.Series(lasso.predict(X_test))
    X_t_lasso=pd.concat([X_t_lasso,X_t])
    y_t=pd.concat([y_t,y_test])
    X_p_lasso.insert(X_p_lasso.shape[1],str(X_p_lasso.shape[1]),lasso.predict(X2))
X_p_lasso.insert(X_p_lasso.shape[1],'mean',X_p_lasso.mean(axis=1))
print(X_t_lasso)
print(y_t)
print(X_p_lasso)

X_t_gbr=pd.Series()
X_p_gbr=pd.DataFrame(index=np.arange(X1.shape[0]))
for train,test in kf.split(X):
    X_train,X_test=X.ix[train,:],X.ix[test,:]
    y_train,y_test=y[train],y[test]
    gbr=GradientBoostingRegressor(loss='ls',n_estimators=300,learning_rate=0.1,max_features=None,max_depth=3,max_leaf_nodes=200)
    gbr.fit(X_train,y_train)
    X_t=pd.Series(gbr.predict(X_test))
    X_t_gbr=pd.concat([X_t_gbr,X_t])
    X_p_gbr.insert(X_p_gbr.shape[1],str(X_p_gbr.shape[1]),gbr.predict(X2))
X_p_gbr.insert(X_p_gbr.shape[1],'mean',X_p_gbr.mean(axis=1))
print(X_t_gbr)
print(X_p_gbr)

input_size=288
hidden_size_1=50
hidden_size_2=10
num_classes=1
num_epochs=2000
Batch_Size=5
learning_rate=0.4

X_t_torch=pd.Series()
X_p_torch=pd.DataFrame(index=np.arange(X1.shape[0]))
for train,test in kf.split(X):
    X_train,X_test=X.ix[train,:],X.ix[test,:]
    y_train,y_test=y[train],y[test]
    train_labels=torch.from_numpy(y_train.values).float()
    train_data=torch.from_numpy(X_train.values).float()
    dev_data=torch.from_numpy(X_test.values).float()
    test_data=torch.from_numpy(X2.values).float()
    train_dataset=Data.TensorDataset(data_tensor=train_data,target_tensor=train_labels)
    dev_dataset=Data.TensorDataset(data_tensor=dev_data,target_tensor=torch.zeros(292,1))
    test_dataset=Data.TensorDataset(data_tensor=test_data,target_tensor=torch.zeros(1459,1))

    train_loader=Data.DataLoader(
        dataset=train_dataset,
        batch_size=Batch_Size,
        shuffle=True,
    )

    dev_loader=Data.DataLoader(
        dataset=dev_dataset,
        batch_size=1,
        shuffle=False,
    )

    test_loader=Data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
    )

    class Net1(nn.Module):
        def __init__(self,input_size,hidden_size_1,num_classes):
            super(Net1,self).__init__()
            self.fc1=nn.Linear(input_size,hidden_size_1)
            self.a=nn.Softplus()
            self.fc2=nn.Linear(hidden_size_1,num_classes)
        
        def forward(self,x):
            out=self.fc1(x)
            out=self.a(out)
            out=self.fc2(out)
            return out

    model=Net1(input_size,hidden_size_1,num_classes)

    criterion=nn.L1Loss()
    optimizer=torch.optim.Adagrad(model.parameters(),lr=learning_rate,weight_decay=0)

    for epoch in range(num_epochs):
        for i,(data,labels) in enumerate(train_loader):
            data=Variable(data)
            labels=Variable(labels).float()

            optimizer.zero_grad()
            outputs=model(data)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

    X_t=pd.Series(index=np.arange(X_test.shape[0]))
    i=0
    for data,labels in dev_loader:
        data=Variable(data)
        outputs=model(data)
        pre=outputs.data.tolist()
        X_t[i]=pre[0][0]
        i+=1
    X_t_torch=pd.concat([X_t_torch,X_t])

    y_p=pd.Series(index=np.arange(test_1.shape[0]))
    i=0
    for data,labels in test_loader:
        data=Variable(data)
        outputs=model(data)
        pre=outputs.data.tolist()
        y_p[i]=pre[0][0]
        i+=1
    X_p_torch.insert(X_p_torch.shape[1],str(X_p_torch.shape[1]),y_p)
X_p_torch.insert(X_p_torch.shape[1],'mean',X_p_torch.mean(axis=1))


X_t=pd.DataFrame(index=np.arange(X.shape[0]))
X_p=pd.DataFrame(index=np.arange(X2.shape[0]))
X_t.insert(X_t.shape[1],'lasso',X_t_lasso)
X_t.insert(X_t.shape[1],'gbr',X_t_gbr)
X_t.insert(X_t.shape[1],'torch',X_t_torch)
X_p.insert(X_p.shape[1],'lasso',X_p_lasso)
X_p.insert(X_p.shape[1],'gbr',X_p_gbr)
X_p.insert(X_p.shape[1],'torch',X_p_torch)

Linear=LinearRegression(fit_intercept=False)
Linear.fit(X_p,y_p)

y_pred=Linear.predict(X_p)

pred=pd.Dataframe(index=X2.shape[0])
pred.insert(0,'Id',test_1['Id'])
pred.insert(1,'SalePrice',y_pred)
pred.to_csv('Price-stacking.csv')
print(time.time()-a)
