# Artificial-Intelligence-assisted-diabetes-genetic-risk-prediction
This is the source code of Tianchi precision medicine competition -- Artificial Intelligence assisted diabetes genetic risk prediction 2018＜/br＞
run main.py＜/br＞
Structure Description＜/br＞
data-＜/br＞
	save the original training and testing data＜/br＞
	save the intermediate files＜/br＞
code-＜/br＞
	addA.py - add the answers of A board＜/br＞
	lgb.py - train model1 based on lightgbm＜/br＞
	XGB_10.py - train ten different models based on xgb-gbdt and blend them to a model2 ＜/br＞
	xgb_dart_f1.py - train model3 based on xgb-dart＜/br＞
	xgb_gbdt_f1.py - train model4 based on xgb-gbdt＜/br＞
	bagging_moedl.py - fusing the different predicting results from above models＜/br＞
submit-＜/br＞
	save final file＜/br＞
