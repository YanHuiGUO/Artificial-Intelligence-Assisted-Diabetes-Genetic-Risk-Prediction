import time
import numpy as np
import pandas as pd
import lightgbm as lgb
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
    int_fea = ['BMI分类','产次','孕次']
    other_fea = [f for f in _predictor_f if (f not in gen_fea) and (f not in int_fea)]
    other_fea.remove('RBP4')
######fill nan############  

    ##mode fill gen_fea
#    _mode_gen_fea =_train_f[gen_fea].mode() 
#    for i in range(len(gen_fea)):
#        _train_f[gen_fea[i]].fillna(_mode_gen_fea[gen_fea[i]][0],inplace = True)
#        _test_f[gen_fea[i]].fillna(_mode_gen_fea[gen_fea[i]][0],inplace = True)
    
    
    #mode fill int_fea
    int_fea = list(set(int_fea))
    _mode_int_fea =_train_f[int_fea].mode() 
    for i in range(len(int_fea)):
        _train_f[int_fea[i]].fillna(_mode_int_fea[int_fea[i]][0],inplace = True)
        _test_f[int_fea[i]].fillna(_mode_int_fea[int_fea[i]][0],inplace = True)
    
    ##median fill other_fea
    _median_other_fea =_train_f[other_fea].median()   
    for i in range(len(other_fea)):
        _train_f[other_fea[i]].fillna(_median_other_fea[other_fea[i]],inplace = True)
        _test_f[other_fea[i]].fillna(_median_other_fea[other_fea[i]],inplace = True)
   
    
    ####not num fea 
#    _train_f['SNP1|SNP2'] = _train_f['SNP1'] ^ _train_f['SNP2']
#    _test_f['SNP1|SNP2'] =  _test_f['SNP1'] ^ _test_f['SNP2']  
    
    
    gen_drop = ['SNP12','SNP26','SNP9','SNP1','SNP10']
#    gen_drop = ['SNP21','SNP22','SNP23','SNP54','SNP55']
    ##onehot gen
#    gen_fea = list(set(gen_fea)-set(gen_drop))
#    gen_fea = gen_fea+['DM家族史']
#    for idx in gen_fea:
#        _le=LabelEncoder().fit(_train_f[idx]) 
#        ##onehot tarin
#        _label=_le.transform(_train_f[idx])
#        _ohe=OneHotEncoder(sparse=False).fit(_label.reshape(-1,1))
#        ohe=_ohe.transform(_label.reshape(-1,1))
#        _train_f.drop([idx],axis=1,inplace=True)
#        for i in range(0,ohe.shape[1]):
#            _train_f[idx+'_'+str(i)] = ohe[:,i]
#        ##onehot test
#        _label=_le.transform(_test_f[idx])
#        ohe=_ohe.transform(_label.reshape(-1,1))
#        _test_f.drop([idx],axis=1,inplace=True)
#        for i in range(0,ohe.shape[1]):
#            _test_f[idx+'_'+str(i)] = ohe[:,i]
################add feature######################### 
    eps=1e-4
    _train_f['ALT/AST'] = _train_f['ALT']/(_train_f['AST']+eps)
    _test_f['ALT/AST'] = _test_f['ALT']/(_test_f['AST']+eps)
    
    _train_f['HDLC/LDLC'] = _train_f['HDLC']/(_train_f['LDLC']+eps)
    _test_f['HDLC/LDLC'] = _test_f['HDLC']/(_test_f['LDLC']+eps)
    
    
    _train_f['CHO/TG'] = _train_f['CHO']/(_train_f['TG']+eps)
    _test_f['CHO/TG'] = _test_f['CHO']/(_test_f['TG']+eps)
    
    _train_f['CHO/Cr'] = _train_f['CHO']/(_train_f['Cr']+eps)
    _test_f['CHO/Cr'] = _test_f['CHO']/(_test_f['Cr']+eps)
    
    
    _train_f['ApoA1/ApoB'] = _train_f['ApoA1']/_train_f['ApoB']
    _test_f['ApoA1/ApoB'] =_test_f['ApoA1']/_test_f['ApoB']
    
    
    _train_f['收缩压/舒张压'] = _train_f['收缩压']/(_train_f['舒张压']+eps)
    _test_f['收缩压/舒张压'] = _test_f['收缩压']/(_test_f['舒张压']+eps)
    
 ### wu da  add 
    _train_f['年龄/HDLC'] = _train_f['年龄']/(_train_f['HDLC']+eps)
    _test_f['年龄/HDLC'] = _test_f['年龄']/(_test_f['HDLC']+eps)
    
    _train_f['AST/TG'] = _train_f['AST']/(_train_f['TG']+eps)
    _test_f['AST/TG'] = _test_f['AST']/(_test_f['TG']+eps)
    
    _train_f['wbc/CHO'] = _train_f['wbc']/(_train_f['CHO']+eps)
    _test_f['wbc/CHO'] = _test_f['wbc']/(_test_f['CHO']+eps)
    
    _train_f['VAR00007/wbc'] = _train_f['VAR00007']/(_train_f['wbc']+eps)
    _test_f['VAR00007/wbc'] = _test_f['VAR00007']/(_test_f['wbc']+eps)   
    
    _train_f['TG/HDLC'] = _train_f['TG']/(_train_f['HDLC']+eps)
    _test_f['TG/HDLC'] = _test_f['TG']/(_test_f['HDLC']+eps)   
    
    _train_f['VAR00007/ApoA1'] = _train_f['VAR00007']/(_train_f['ApoA1']+eps)
    _test_f['VAR00007/ApoA1'] = _test_f['VAR00007']/(_test_f['ApoA1']+eps)   
    
   ############################## 
    _train_f['VAR00007log年龄'] = np.log10(_train_f['VAR00007'])*np.log10(_train_f['年龄']/(1e-5))
    _test_f['VAR00007log年龄'] =np.log10(_test_f['VAR00007'])*np.log10(_test_f['年龄']/(1e-5))


#    _train_f['ApoA1/年龄'] = _train_f['ApoA1']/_train_f['年龄']
#    _test_f['ApoA1/年龄'] =_test_f['ApoA1']/_test_f['年龄']
#
    
    _train_f['体脂率'] = 1.2*_train_f['孕前BMI']+0.23*_train_f['年龄']-5.4
    _test_f['体脂率'] =  1.2*_test_f['孕前BMI']+0.23*_test_f['年龄']-5.4
   
    _train_f['VAR00007*孕前BMI'] = _train_f['VAR00007']*(_train_f['孕前BMI']+eps)
    _test_f['VAR00007*孕前BMI'] = _test_f['VAR00007']*(_test_f['孕前BMI']+eps)  
    add_fea = ['ALT/AST','HDLC/LDLC','CHO/TG','CHO/Cr',\
               'ApoA1/ApoB',\
               '体脂率','收缩压/舒张压','VAR00007log年龄','年龄/HDLC',\
               'AST/TG','wbc/CHO','VAR00007/wbc','TG/HDLC','VAR00007/ApoA1','VAR00007*孕前BMI']
    
    minmax_fea = other_fea+add_fea
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(_train_f[minmax_fea])
    _train_f[minmax_fea]=min_max_scaler.transform(_train_f[minmax_fea])
    _test_f[minmax_fea]=min_max_scaler.transform(_test_f[minmax_fea])
    
###########drop spme useless########## 
    _train_f.drop(['BMI分类'],axis = 1,inplace = True)
    _test_f.drop(['BMI分类'],axis = 1,inplace = True)
    
    _train_f.drop(['产次'],axis = 1,inplace = True)
    _test_f.drop(['产次'],axis = 1,inplace = True)
    
    _train_f.drop(['身高'],axis = 1,inplace = True)
    _test_f.drop(['身高'],axis = 1,inplace = True)

    _train_f.drop(['孕前体重'],axis = 1,inplace = True)
    _test_f.drop(['孕前体重'],axis = 1,inplace = True)

    _train_f.drop(['糖筛孕周'],axis = 1,inplace = True)
    _test_f.drop(['糖筛孕周'],axis = 1,inplace = True)

#    _train_f.drop(['孕次'],axis = 1,inplace = True)
#    _test_f.drop(['孕次'],axis = 1,inplace = True)
    

#    _train_f.drop(['RBP4'],axis = 1,inplace = True)
#    _test_f.drop(['RBP4'],axis = 1,inplace = True)
    
    ###gen drop 

    _train_f.drop(gen_drop,axis = 1,inplace = True)
    _test_f.drop(gen_drop,axis = 1,inplace = True)
    
###########drop spme useless##########   
    _train_f.drop(['id'],axis = 1,inplace = True)
    _test_f.drop(['id'],axis = 1,inplace = True)
    

    return _train_f,_test_f
 
#%%
orig_train = pd.read_csv('../data/d_train_20180307.csv',encoding='gb2312')
orig_test = pd.read_csv('../data/f_test_b_20180305.csv',encoding='gb2312')


predictor = [f for f in orig_train.columns if f not in ['label']]
train,test = deal_fea(orig_train.copy(),orig_test.copy(),predictor)

predictor = [f for f in train.columns if f not in ['label']]



#train.to_csv('off_train_fea.csv',index=False)
#test.to_csv('off_test_fea.csv',index=False)
#%%
print('开始训练...')
predictor = [f for f in train.columns if f not in ['label']]
xgb_params={'booster':'dart',
        	'objective': 'binary:logistic',
        	'max_depth':8,
        	'lambda':10,
        	'subsample':0.6,
        	'colsample_bytree':0.6,
        	'eta': 0.05, #0.01
        	'seed':1024,
        	'nthread':12,
            'n_estimators':70,
            'min_child_weight':5,
            'gamma':0.1
        	}
#cv_params= {'n_estimators': np.array(range(10,200,20)), \
#            'eta': [0.001,0.01,0.1,0.2],\
#            'max_depth': [6,7,8,9,10],\
#            'lambda': [10,30,60,100],\
#            'gamma':[0.1,0.3,0.5,0.8]
#            }
#score = 'f1'
#optimized_GBM = GridSearchCV(xgb.XGBClassifier(**xgb_params),cv_params,cv=5,  verbose=2,scoring=score)
#
#optimized_GBM.fit(train[predictor], train['label'])
#optimized_GBM.best_params_
#%%
def f1_error(preds,df):
    label = df.get_label()
#    preds = 1.0/(1.0+np.exp(-preds))
    pred = preds.copy()
    pred[preds>=0.5]=1
    pred[preds<0.5]=0
    tp = sum([int (i==1 and j==1) for i,j in zip(pred,label)])
    precision = float (tp)/sum(pred==1)
    recall = float(tp)/sum(label==1)
    return 'f1',2*(precision*recall/(precision+recall))
def check_f1(preds,gt):
    label = gt.copy()
    pred = preds.copy()
    pred[preds>=0.5]=1
    pred[preds<0.5]=0
    tp = sum([int (i==1 and j==1) for i,j in zip(pred,label)])
    precision = float (tp)/sum(pred==1)
    recall = float(tp)/sum(label==1)
    return 2*(precision*recall/(precision+recall))
print('开始CV 5折训练...')
t0 = time.time()
every_loop_num=5
loop_num = 1
train_preds = np.zeros((train.shape[0],loop_num))
scores =  np.zeros(loop_num)
test_preds = np.zeros((test.shape[0],every_loop_num*loop_num))

for loop in range(loop_num):
    kf = StratifiedKFold(train.label,n_folds = every_loop_num, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):
        print('第{}-{}次训练...'.format(loop,i))
        train_feat1 = train.iloc[train_index].copy()
        train_feat2 = train.iloc[test_index].copy()
        
        xgb_train1 = xgb.DMatrix(train_feat1[predictor], train_feat1['label'])
        xgb_train2 = xgb.DMatrix(train_feat2[predictor], train_feat2['label'] )
        watchlist  = [(xgb_train1,'train'),(xgb_train2,'test')]
        cv_log = xgb.cv(xgb_params,xgb_train1,num_boost_round=25000,nfold=5,feval=f1_error,\
                    early_stopping_rounds=50,seed=1024,maximize=True)
        bst_auc= cv_log['test-f1-mean'].max()
        cv_log['best'] = cv_log.index
        cv_log.index = cv_log['test-f1-mean']
        bst_nb = cv_log.best.to_dict()[bst_auc]
        
        #train bst_nb+50
        watchlist  = [(xgb_train1,'train')]
        xgb_model = xgb.train(xgb_params,xgb_train1,num_boost_round=bst_nb+50,evals=watchlist)
      
        xgb_cv_test= xgb.DMatrix(train_feat2[predictor])
        
       
        xgb_pre  = xgb_model.predict(xgb_cv_test)
        train_preds[test_index,loop] = xgb_pre
        
        
        xgb_Dmatrix= xgb.DMatrix(test[predictor])
        
        xgb_pre_test = xgb_model.predict(xgb_Dmatrix)
        test_preds[:,i+loop*every_loop_num] = xgb_pre_test

    
    print('线下train得分：    {}'.format(check_f1(train_preds[:,loop],train['label'])))
    scores[loop]=check_f1(train_preds[:,loop],train['label'])
#%%
submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
pre = submission['pred']
print('线下train_avg得分：  {}'.format(np.mean(scores)))

pre.to_csv('../data/xgb_dart_f1.csv',index=False, float_format='%f')