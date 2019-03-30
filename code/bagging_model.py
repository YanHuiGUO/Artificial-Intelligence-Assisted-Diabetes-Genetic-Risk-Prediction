#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:57:32 2018

@author: root
"""

import numpy as np
import pandas as pd
from sklearn import metrics
def check_f1(preds,gt):
    label = gt.copy()
    pred = preds.copy()
    pred[preds>=0.5]=1
    pred[preds<0.5]=0
#    tp = sum([int (i==1 and j==1) for i,j in zip(pred,label)])
#    precision = float (tp)/sum(pred==1)
#    recall = float(tp)/sum(label==1)
#    return 2*(precision*recall/(precision+recall))
    return metrics.f1_score(label,pred)

lgb_f1 = pd.read_csv('../data/lgb_f1.csv',header=None)
xgb_dart_f1 = pd.read_csv('../data/xgb_dart_f1.csv',header=None)
xgb_gbdt_f1 = pd.read_csv('../data/xgb_gbdt_f1.csv',header=None)


train =  pd.read_csv('../data/d_train_20180307.csv',encoding='gb2312')
orig_test =  pd.read_csv('../data/f_test_b_20180305.csv',encoding='gb2312')
ans=pd.read_csv('../data/xgb0_f2_new.csv')

result = pd.concat([
                    xgb_dart_f1,\
                    xgb_gbdt_f1,\
                    lgb_f1
                    ],axis=1,ignore_index=True)
res = np.zeros_like(lgb_f1)
bag = pd.DataFrame({0:result.mean(axis=1)})
res[np.where(bag[0]>0.47)[0]]=1  #0.47
res[np.where(bag[0]<=0.47)[0]]=0

##post deal
change = np.where((res==1)&(ans==0))[0]
ans.loc[change] = 1

submission = pd.DataFrame()
submission['pred']=np.reshape(ans['pred'],-1,1)

import datetime
submission.to_csv(("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"),float_format='%d', header=None, index=False)
