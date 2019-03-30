# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 13:41:20 2018

@author: HUST
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:25:36 2018

@author: HUST
"""
import time
import random
import operator
import numpy as np
import pandas as pd
# import lightgbm as lgb
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import LabelEncoder 
from sklearn import preprocessing
##
#%%
def deal_fea(_train_f,_test_f,_predictor_f):

    
    gen_fea = [f for f in _predictor_f if 'SNP' in f]
#    int_fea = ['BMI分类','产次','孕次']
    gen_fea.extend(['ACEID','DM家族史'])
    other_fea = [f for f in _predictor_f if (f not in gen_fea)]
#    other_fea.remove('RBP4')
######fill nan############  

    
    ##mode fill gen_fea
    _mode_gen_fea =_train_f[gen_fea].mode() 
    for i in range(len(gen_fea)):
        _train_f[gen_fea[i]].fillna(0,inplace = True)
        _test_f[gen_fea[i]].fillna(0,inplace = True)
    
    
    #mode fill int_fea
#    int_fea = list(set(int_fea))
#    _mode_int_fea =_train_f[int_fea].mode()
#    _mode_int_fea_test =_test_f[int_fea].mode()
#    for i in range(len(int_fea)):
#        _train_f[int_fea[i]].fillna(_mode_int_fea[int_fea[i]][0],inplace = True)
#        _test_f[int_fea[i]].fillna(_mode_int_fea_test[int_fea[i]][0],inplace = True)
#    
    ##median fill other_fea
    _median_other_fea =_train_f[other_fea].median() 
    _median_other_fea_test =_test_f[other_fea].median()
    for i in range(len(other_fea)):
        _train_f[other_fea[i]].fillna(_median_other_fea[other_fea[i]],inplace = True)
        _test_f[other_fea[i]].fillna(_median_other_fea_test[other_fea[i]],inplace = True)
   
    
    ####not num fea 
#    _train_f['SNP1|SNP2'] = _train_f['SNP1'] ^ _train_f['SNP2']
#    _test_f['SNP1|SNP2'] =  _test_f['SNP1'] ^ _test_f['SNP2']  
    
    
#    gen_drop = ['SNP12','SNP26','SNP9','SNP1','SNP10']
#    gen_drop = ['SNP21','SNP22','SNP23','SNP54','SNP55']
    #onehot gen
#    gen_fea = list(set(gen_fea)-set(gen_drop))
    
#    other_fea.remove('DM家族史')
    for idx in gen_fea:
        _le=LabelEncoder().fit(_train_f[idx]) 
        ##onehot tarin
        _label=_le.transform(_train_f[idx])
        _ohe=OneHotEncoder(sparse=False).fit(_label.reshape(-1,1))
        ohe=_ohe.transform(_label.reshape(-1,1))
        _train_f.drop([idx],axis=1,inplace=True)
        for i in range(0,ohe.shape[1]):
            _train_f[idx+'_'+str(i)] = ohe[:,i]
        ##onehot test
        _le_test=LabelEncoder().fit(_test_f[idx])
        _label_test=_le_test.transform(_test_f[idx])
        _ohe_test=OneHotEncoder(sparse=False).fit(_label_test.reshape(-1,1))
        ohe_test=_ohe_test.transform(_label_test.reshape(-1,1))
        _test_f.drop([idx],axis=1,inplace=True)
        for i in range(0,ohe_test.shape[1]):
            _test_f[idx+'_'+str(i)] = ohe_test[:,i]
################add feature######################### 
#    eps=1e-4
#    _train_f['ALT/AST'] = _train_f['ALT']/(_train_f['AST']+eps)
#    _test_f['ALT/AST'] = _test_f['ALT']/(_test_f['AST']+eps)
#    
#    _train_f['HDLC/LDLC'] = _train_f['HDLC']/(_train_f['LDLC']+eps)
#    _test_f['HDLC/LDLC'] = _test_f['HDLC']/(_test_f['LDLC']+eps)
#    
#    
#    _train_f['CHO/TG'] = _train_f['CHO']/(_train_f['TG']+eps)
#    _test_f['CHO/TG'] = _test_f['CHO']/(_test_f['TG']+eps)
#    
#    _train_f['CHO/Cr'] = _train_f['CHO']/(_train_f['Cr']+eps)
#    _test_f['CHO/Cr'] = _test_f['CHO']/(_test_f['Cr']+eps)
#    
#    
#    _train_f['ApoA1/ApoB'] = _train_f['ApoA1']/_train_f['ApoB']
#    _test_f['ApoA1/ApoB'] =_test_f['ApoA1']/_test_f['ApoB']
#    
#    
#    _train_f['收缩压/舒张压'] = _train_f['收缩压']/(_train_f['舒张压']+eps)
#    _test_f['收缩压/舒张压'] = _test_f['收缩压']/(_test_f['舒张压']+eps)
#    
# ### wu da  add 
#    _train_f['年龄/HDLC'] = _train_f['年龄']/(_train_f['HDLC']+eps)
#    _test_f['年龄/HDLC'] = _test_f['年龄']/(_test_f['HDLC']+eps)
#    
#    _train_f['AST/TG'] = _train_f['AST']/(_train_f['TG']+eps)
#    _test_f['AST/TG'] = _test_f['AST']/(_test_f['TG']+eps)
#    
#    _train_f['wbc/CHO'] = _train_f['wbc']/(_train_f['CHO']+eps)
#    _test_f['wbc/CHO'] = _test_f['wbc']/(_test_f['CHO']+eps)
#    
#    _train_f['VAR00007/wbc'] = _train_f['VAR00007']/(_train_f['wbc']+eps)
#    _test_f['VAR00007/wbc'] = _test_f['VAR00007']/(_test_f['wbc']+eps)   
#    
#    _train_f['TG/HDLC'] = _train_f['TG']/(_train_f['HDLC']+eps)
#    _test_f['TG/HDLC'] = _test_f['TG']/(_test_f['HDLC']+eps)   
#    
#    _train_f['VAR00007/ApoA1'] = _train_f['VAR00007']/(_train_f['ApoA1']+eps)
#    _test_f['VAR00007/ApoA1'] = _test_f['VAR00007']/(_test_f['ApoA1']+eps)   
#    
#   ############################## 
#    _train_f['VAR00007log年龄'] = np.log10(_train_f['VAR00007'])*np.log10(_train_f['年龄']/(1e-5))
#    _test_f['VAR00007log年龄'] =np.log10(_test_f['VAR00007'])*np.log10(_test_f['年龄']/(1e-5))


#    _train_f['ApoA1/年龄'] = _train_f['ApoA1']/_train_f['年龄']
#    _test_f['ApoA1/年龄'] =_test_f['ApoA1']/_test_f['年龄']
    _train_f['VAR00007*BMI'] = _train_f['VAR00007']*_train_f['孕前BMI']
    _test_f['VAR00007*BMI'] = _test_f['VAR00007']*_test_f['孕前BMI']
#    _train_f['体脂率'] = 1.2*_train_f['孕前BMI']+0.23*_train_f['年龄']-5.4
#    _test_f['体脂率'] =  1.2*_test_f['孕前BMI']+0.23*_test_f['年龄']-5.4

#    add_fea = ['ALT/AST','HDLC/LDLC','CHO/TG','CHO/Cr',\
#               'ApoA1/ApoB',\
#               '体脂率','收缩压/舒张压','VAR00007log年龄','年龄/HDLC',\
#               'AST/TG','wbc/CHO','VAR00007/wbc','TG/HDLC','VAR00007/ApoA1']
#    
#    minmax_fea = other_fea+add_fea
#    min_max_scaler = preprocessing.MinMaxScaler()
#    min_max_scaler.fit(_train_f[minmax_fea])
#    _train_f[minmax_fea]=min_max_scaler.transform(_train_f[minmax_fea])
#    _test_f[minmax_fea]=min_max_scaler.transform(_test_f[minmax_fea])
    
###########drop spme useless########## 
#    _train_f.drop(['BMI分类'],axis = 1,inplace = True)
#    _test_f.drop(['BMI分类'],axis = 1,inplace = True)
#    
#    _train_f.drop(['产次'],axis = 1,inplace = True)
#    _test_f.drop(['产次'],axis = 1,inplace = True)
#    
#    _train_f.drop(['身高'],axis = 1,inplace = True)
#    _test_f.drop(['身高'],axis = 1,inplace = True)
#
#    _train_f.drop(['孕前体重'],axis = 1,inplace = True)
#    _test_f.drop(['孕前体重'],axis = 1,inplace = True)
#
#    _train_f.drop(['糖筛孕周'],axis = 1,inplace = True)
#    _test_f.drop(['糖筛孕周'],axis = 1,inplace = True)
#
##    _train_f.drop(['孕次'],axis = 1,inplace = True)
#    _test_f.drop(['孕次'],axis = 1,inplace = True)
    

#    _train_f.drop(['RBP4'],axis = 1,inplace = True)
#    _test_f.drop(['RBP4'],axis = 1,inplace = True)
    
    ###gen drop 

#    _train_f.drop(gen_drop,axis = 1,inplace = True)
#    _test_f.drop(gen_drop,axis = 1,inplace = True)
    
###########drop spme useless##########   
    _train_f.drop(['id'],axis = 1,inplace = True)
    _test_f.drop(['id'],axis = 1,inplace = True)
    

    return _train_f,_test_f
 
#%%
orig_test = pd.read_csv('../data/f_test_b_20180305.csv',encoding='gb2312')
orig_train = pd.read_csv('../data/f_train_20180204.csv',encoding='gb2312')
orig_test1 = pd.read_csv('../data/f_test_a_20180204.csv',encoding='gb2312')
GT = pd.read_csv('../data/f_answer_a_20180306.csv', header=None)
orig_test1['label'] = GT
orig_train = pd.concat([orig_train, orig_test1], axis=0, ignore_index=True)
#orig_train = pd.read_csv('val_train.csv',encoding='gb2312')
#orig_test = pd.read_csv('val_test.csv',encoding='gb2312')
#test_y = orig_test['label']
#orig_test = orig_test.drop(['label'], axis=1)

predictor = [f for f in orig_train.columns if f not in ['label']]
train,test = deal_fea(orig_train.copy(),orig_test.copy(),predictor)
# train = train.sample(frac = 1)#打乱数据
predictor = list(set(train.columns)&set(test.columns))
#predictor = [f for f in train.columns if f not in ['label']]

def f1_val(preds,df):
    label = df.get_label()
    pred = preds.copy()
    pred[preds>=0.5]=1
    pred[preds<0.5]=0
    tp = sum([int (i==1 and j==1) for i,j in zip(pred,label)])
    precision = float (tp)/sum(pred==1)
    recall = float(tp)/sum(label==1)
    return ('f1',-2*(precision*recall/(precision+recall)))


def f1_error(preds,df):
    label = df.get_label().values.copy()
    pred = preds.copy()
    pred[preds>=0.5]=1
    pred[preds<0.5]=0
    tp = sum([int (i==1 and j==1) for i,j in zip(pred,label)])
    precision = float (tp)/sum(pred==1)
    recall = float(tp)/sum(label==1)
    return ('f1',-2*(precision*recall/(precision+recall)))

def check_f1(preds,gt):
    label = gt.copy()
    pred = preds.copy()
    pred[preds>=0.5]=1
    pred[preds<0.5]=0
    tp = sum([int (i==1 and j==1) for i,j in zip(pred,label)])
    precision = float (tp)/sum(pred==1)
    recall = float(tp)/sum(label==1)
    return 2*(precision*recall/(precision+recall))

def pre_f1(preds,gt):
    label = gt.copy()
    pred = preds.copy()
#    pred[preds>=0.44]=1
#    pred[preds<0.44]=0
    tp = sum([int (i==1 and j==1) for i,j in zip(pred,label)])
    precision = float (tp)/sum(pred==1)
    recall = float(tp)/sum(label==1)
    return 2*(precision*recall/(precision+recall))

print('the select feature is {0}'.format(len(predictor)))
def pipeline(train,iteration,random_seed,max_depth,lambd,subsample,colsample_bytree,min_child_weight,n_feature,nround):
    train_y = train['label']
    dtrain = xgb.DMatrix(train[predictor],label=train_y)
    dtest = xgb.DMatrix(test[predictor])

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
#        'scale_pos_weight':float(len(train_y)-sum(train_y))/sum(train_y),
        'eval_metric': 'auc',
        'max_depth': 4,
        'lambda': 0,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'eta': 0.01,
        'seed': random_seed,
        'nthread':4
    }

    watchlist  = [(dtrain,'train')]
    cv_log = xgb.cv(params,dtrain,num_boost_round=25000,nfold=5,metrics='auc', verbose_eval=50,early_stopping_rounds=50,seed=1024)#metrics='auc',
    bst_auc= cv_log['test-auc-mean'].max()
    cv_log['best'] = cv_log.index
    cv_log.index = cv_log['test-auc-mean']
    bst_nb = cv_log.best.to_dict()[bst_auc]
    print(bst_nb)
    
    #train bst_nb+50
    watchlist  = [(dtrain,'train')]
    xgb_model = xgb.train(params,dtrain,num_boost_round=bst_nb+50,evals=watchlist)#,
    
    importance = xgb_model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    
    xgb_test = xgb_model.predict(dtest)
    
    return xgb_test

if __name__ == '__main__':    
    random_seed = list(range(10))
    max_depth = [4, 5]
    lambd = list(range(50, 150))
    subsample = [i / 1000.0 for i in range(700, 800, 10)]
    colsample_bytree = [i / 1000.0 for i in range(700, 800, 10)]
    min_child_weight = [i / 100.0 for i in range(150, 250, 10)]
    n_feature = [i / 100.0 for i in range(1, 80)]
    nround = list(range(450,550,5)) # xgboost: the iteration numbers

#    random.shuffle(random_seed)
#    random.shuffle(max_depth)
#    random.shuffle(lambd)
#    random.shuffle(subsample)
#    random.shuffle(colsample_bytree)
#    random.shuffle(min_child_weight)
#    random.shuffle(n_feature)
#    random.shuffle(nround)
    scores =  np.zeros(10)
    test_preds = np.zeros((test.shape[0],10))
    for i in range(10):
        print("iter:{}".format(i))
        xgb_pre_test = pipeline(train,i,random_seed[i%len(random_seed)],max_depth[i%2],lambd[i%len(lambd)],\
                                                    subsample[i%len(subsample)],colsample_bytree[i%len(colsample_bytree)],\
                                                    min_child_weight[i%len(min_child_weight)],n_feature[i%len(n_feature)],nround[i%len(nround)])
        test_preds[:,i] = xgb_pre_test
#        scores[i]=check_f1(test_preds[:,i],GT['pred'])
#        print('线下得分：  {}'.format(scores[i]))
    pred = test_preds.mean(axis=1)
    # pre = submission['pred'].apply(lambda x: 1 if x>=0.5 else 0)
    pre = pd.DataFrame({'pred':pred})
    pre.to_csv('../data/xgb0_pro.csv',index=False)
    
    
    threshold = list(pre['pred'])
    threshold.sort(reverse=True)
    thres = threshold[int(200*0.47)]
    pre['pred'] = pre['pred'].apply(lambda x: 1 if x>=thres else 0)
    pre.loc[66]=1
    pre.loc[129]=0
    pre.loc[140]=1
    pre.loc[195]=0
    pre.loc[95]=0
#    F1_score = pre_f1( pre['pred'],GT['pred'])
#    print('线上得分：    {}'.format(F1_score))
    print('线上正样本个数：    {}'.format(sum(pre['pred'])))
#    score = test_y - pre['pred']
#    score = score.apply(lambda x: 1 if x==0 else 0)
#    print('线下val得分：  {}'.format(sum(score)/len(score)))
    pre.to_csv('../data/xgb0_f2_new.csv',index=False)