#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
test=pd.read_csv('test.csv')
train=pd.read_csv('train_2.csv')
del train[list(train)[0]]
#print(train.ix[0:train.shape[0],0])

test_1=pd.DataFrame(index=np.arange(test.shape[0]))
test_2=pd.DataFrame(index=np.arange(test.shape[0]))

for i in range(test.shape[1]):
    print(i)
    print(test.ix[[],i].dtype)
    if test.ix[[],i].dtype==object:
        mclass=[]
        for j in test.ix[0:test.shape[0],i]:
            if j not in mclass and j==j:
                 mclass.append(j)
        for classes in mclass:
            test_1[list(test)[i]+'_'+str(classes)]=0
        for j in range(test.shape[0]):
            for classes in mclass:
                if test.ix[j,i]==str(classes):
                    test_1.ix[j,list(test)[i]+'_'+str(classes)]=1
        print(mclass)
    else:
        test_2.insert(test_2.shape[1],list(test)[i],test.ix[0:test.shape[0],i])
        if test.ix[[],i].dtype=='float64':
            m=test.mean()[list(test)[i]]
            for j in range(test.shape[0]):
                if test.ix[j,i]!=test.ix[j,i]:
                    test_2.ix[j,list(test)[i]]=m
            print(m)
test_1.insert(0,'Id',test['Id'])
print(test_1)
print(test_2)
test_1.to_csv('test_2.csv')
test_2.to_csv('test_3.csv')

