# Artificial-Intelligence-assisted-diabetes-genetic-risk-prediction
This is the source code of Tianchi precision medicine competition -- Artificial Intelligence assisted diabetes genetic risk prediction 2018  
  *run main.py  
  *Structure Description  
  *data-  
   *save the original training and testing data  
   *save the intermediate files  
  *code-  
	addA.py - add the answers of A board  
	lgb.py - train model1 based on lightgbm  
	XGB_10.py - train ten different models based on xgb-gbdt and blend them to a model2   
	xgb_dart_f1.py - train model3 based on xgb-dart  
	xgb_gbdt_f1.py - train model4 based on xgb-gbdt  
	bagging_moedl.py - fusing the different predicting results from above models  
submit-  
	save final file  
