#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import math
train=pd.read_csv('train.csv')

train_1=pd.DataFrame(columns=list(train),index=['type','mean','var'])

for i in range(train.shape[1]):
    print(i)
    print(train.ix[[],i].dtype)
    #print(train.ix[[],i])
    if train.ix[[],i].dtype==object:
        train_1.ix[0,i]='class'
        mclass=[]
        for j in range(train.shape[0]):
            if train.ix[j,i]!=train.ix[j,i]:
                train.ix[j,i]='nan'
            if train.ix[j,i] not in mclass:
                 mclass.append(train.ix[j,i])
        info=0
        for classes in mclass:
            n=0
            for j in train.ix[0:train.shape[0],i]:
                if j==classes:
                    n+=1
            print(classes)
            print(n)
            n=n/train.shape[0]
            info-=math.log(n,2)*n
        train_1.ix[1,i]=info
    else:
        train_1.ix[0,i]='num'
        m=train.mean()[list(train)[i]]
        train_1.ix[1,i]=m
        if train.ix[[],i].dtype=='float64':
            for j in range(train.shape[0]):
                if train.ix[j,i]!=train.ix[j,i]:
                    train.ix[j,list(train)[i]]=m
        train_1.ix[2,i]=train.var()[list(train)[i]]
    print(train_1)
train_1.to_csv('train_a.csv')
