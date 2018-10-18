#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
train=pd.read_csv('train.csv')

#print(train.ix[0:train.shape[0],0])

train_1=pd.DataFrame(index=np.arange(train.shape[0]))
#train_2=pd.DataFrame(index=['type','mean','var'])
for i in range(train.shape[1]):
    print(i)
    print(train.ix[[],i].dtype)
    print(train.ix[[],i])
    if train.ix[[],i].dtype==object:
        mclass=[]
        for j in train.ix[0:train.shape[0],i]:
            if j not in mclass and j==j:
                 mclass.append(j)
        for classes in mclass:
            train_1[list(train)[i]+'_'+str(classes)]=0
        for j in range(train.shape[0]):
            for classes in mclass:
                if train.ix[j,i]==str(classes):
                    train_1.ix[j,list(train)[i]+'_'+str(classes)]=1
        print(mclass)
    else:
        train_1.insert(train_1.shape[1],list(train)[i],train.ix[:train.shape[0],i])
        train_2.ix[0,list(train)[i]]='b'
        if train.ix[[],i].dtype=='float64':
            m=train.mean()[list(train)[i]]
            for j in range(train.shape[0]):
                if train.ix[j,i]!=train.ix[j,i]:
                    train_1.ix[j,list(train)[i]]=m
            print(m)
print(train_1)
train_1.to_csv('train_0.csv')
