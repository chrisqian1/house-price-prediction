#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(1)

a=time.time()
input_size=130
hidden_size_1=50
hidden_size_2=10
num_classes=1
num_epochs=4500
Batch_Size=5
learning_rate=0.5

train=pd.read_csv('train_lasso.csv')
test=pd.read_csv('test_lasso.csv')
del train[list(train)[0]]
del test[list(test)[0]]

X=train.ix[0:train.shape[0],1:train.shape[1]-1]
y=train['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
X1=test.ix[0:test.shape[0],1:test.shape[1]]
X2=X1.reindex(columns=list(X)).fillna(0)

train_labels=torch.from_numpy(y_train.values).float()
train_data=torch.from_numpy(X_train.values).float()
dev_labels=torch.from_numpy(y_test.values).float()
dev_data=torch.from_numpy(X_test.values).float()
test_data=torch.from_numpy(X2.values).float()

train_dataset=Data.TensorDataset(data_tensor=train_data,target_tensor=train_labels)
dev_dataset=Data.TensorDataset(data_tensor=dev_data,target_tensor=dev_labels)
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
        #print(i)
        #print(model.state_dict())
        #print(outputs)
        #print(labels)
        #if(epoch>900 and i<10):
        #    print(i)
        #    print(outputs)
        #    print(labels)
        #    print(loss)
        loss.backward()
        optimizer.step()

train_loader=Data.DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=False,
)

sum_m=0
for data,labels in train_loader:
    data=Variable(data)
    outputs=model(data)
    #print(outputs)
    #print(labels)
    sum_m+=(math.log(outputs)-math.log(labels))**2
sum_e=np.sqrt(sum_m/len(X_train))
print(sum_e)

sum_mean=0
for data,labels in dev_loader:
    data=Variable(data)
    outputs=model(data)
    #print(outputs)
    #print(labels)
    sum_mean+=(math.log(outputs)-math.log(labels))**2
sum_erro=np.sqrt(sum_mean/len(X_test))
print(sum_erro)

series=pd.DataFrame(index=np.arange(test.shape[0]),columns=['Saleprice'])
i=0
for data,labels in test_loader:
    data=Variable(data)
    outputs=model(data)
    pred=outputs.data.tolist()
    series.ix[i,0]=pred[0][0]
    i+=1
pred=pd.DataFrame(index=np.arange(test.shape[0]))
pred.insert(0,'Id',test['Id'])
pred.insert(1,'SalePrice',series['Saleprice'])

pred.to_csv('Price-torch-2.csv')
print(time.time()-a)
