import numpy as np
import pandas as pd
import lightgbm
from collections import Counter
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc,roc_auc_score,recall_score,precision_score,plot_roc_curve, f1_score
from lightgbm import LGBMClassifier

training = pd.read_csv(training_file)
array_training = training.values
X_training = array_training[:,1:feaure_number]
y_training = array_training[:,0]
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.25, random_state = 666, stratify=y_training)

def Find_Optimal_Cutoff(TPR, FPR, threshold):
	y = TPR - FPR
	Youden_index = np.argmax(y)
	optimal_threshold = threshold[Youden_index]
	point = [FPR[Youden_index], TPR[Youden_index]]
	return optimal_threshold, point

def ROC(label, y_prob):
	fpr, tpr, thresholds = roc_curve(label, y_prob, pos_label=1)
	roc_auc = auc(fpr, tpr)
	optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
	return fpr, tpr, roc_auc, optimal_th, optimal_point

def Evaluation_matrix(row):
	group = row["group"]
	predict = row["predict_group"]
	if group == 0 and predict == 0:
		return "TN"
	if group == 0 and predict == 1:
		return "FP"
	if group == 1 and predict == 1:
		return "TP"
	if group == 1 and predict == 0:
		return "FN"
  
def calculate_MCC(TP_number,FN_number,FP_number,TN_number):
	AA = TP_number
	BB = FN_number
	CC = FP_number
	DD = TN_number
	try:
		Accuracy = (AA+DD)/(AA+BB+CC+DD)
	except ZeroDivisionError:
		Accuracy = float('nan')
	try:
		precision = AA / (AA + CC)
	except ZeroDivisionError:
		precision = float('nan')
	try:
		NPV = DD / (BB + DD)
	except ZeroDivisionError:
		NPV = float('nan')
	try:
		Sensitivity = AA / (AA + BB)
	except ZeroDivisionError:
		Sensitivity= float('nan')
	try:
		Specificity = DD / (CC + DD)
	except ZeroDivisionError:
		Specificity = float('nan')
	try:
		F1 = 2 * precision * Sensitivity / (precision + Sensitivity)
	except ZeroDivisionError:
		F1 = float('nan')
	try:
		numerator = (AA * DD) - (CC * BB)
		denominator = sqrt((AA + CC) * (AA + BB) * (DD + CC) * (DD + BB))
		MCC = numerator/denominator
	except ZeroDivisionError:
		MCC = float('nan')
	return Accuracy, precision, NPV, Sensitivity, Specificity, F1, MCC 

def machine_learning(n_estimators_option,learning_rate_option,num_leaves_option, max_depth_option,min_child_samples_option, reg_alpha_option,reg_lambda_option):
	machine_model = lightgbm.LGBMClassifier(boosting_type='gbdt', n_estimators=n_estimators_option, learning_rate=learning_rate_option, num_leaves=num_leaves_option, 
		                                   max_depth=max_depth_option, min_child_samples=min_child_samples_option,reg_alpha=reg_alpha_option, reg_lambda=reg_lambda_option,
		                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,objective='binary',metrics='auc',class_weight='balanced',random_state=666)
	machine_model.fit(X_train, y_train)
	
	# cross_val_score
	AUC_cross_score = cross_val_score(machine_model,X_train, y_train,cv=10, scoring='roc_auc').mean()

	# probability_score
	train_file = pd.DataFrame(machine_model.predict_proba(X_train))
	train_file.columns =["negative_score", "pos_score"]
	test_file = pd.DataFrame(machine_model.predict_proba(X_test))
	test_file.columns =["negative_score", "pos_score"]
	training_file = pd.DataFrame(machine_model.predict_proba(X_training))
	training_file.columns =["negative_score", "pos_score"]

	# true_label
	train_file['true_label'] = y_train
	test_file['true_label'] = y_test
	training_file['true_label'] = y_training
	training_file_labels = training_file['true_label']
	training_file_preds = training_file['pos_score']
	training_fpr, training_tpr, training_roc_auc, training_optimal_th, training_optimal_point = ROC(training_file_labels, training_file_preds)

	AAAAA = training_optimal_th
	train_file['predict_group'] = train_file['pos_score'].map(lambda x: 1 if x >= AAAAA else 0)
	train_file['group'] = train_file['true_label']
	train_file['result'] = train_file[['group', 'predict_group']].apply(Evaluation_matrix, axis=1)
	c_train = Counter(train_file['result'])
	train_AA = c_train["TP"]
	train_BB = c_train["FN"]
	train_CC = c_train["FP"]
	train_DD = c_train["TN"]
	train_Accuracy, train_precision, train_NPV, train_Sensitivity, train_Specificity, train_F1, train_MCC = calculate_MCC(train_AA,train_BB,train_CC,train_DD)

	test_file['predict_group'] = test_file['pos_score'].map(lambda x: 1 if x >= AAAAA else 0)
	test_file['group'] = test_file['true_label']
	test_file['result'] = test_file[['group', 'predict_group']].apply(Evaluation_matrix, axis=1)
	c_test = Counter(test_file['result'])
	test_AA = c_test["TP"]
	test_BB = c_test["FN"]
	test_CC = c_test["FP"]
	test_DD = c_test["TN"]
	test_Accuracy, test_precision, test_NPV, test_Sensitivity, test_Specificity, test_F1, test_MCC = calculate_MCC(test_AA, test_BB, test_CC, test_DD)

	training_file['predict_group'] = training_file['pos_score'].map(lambda x: 1 if x >= AAAAA else 0)
	training_file['group'] = training_file['true_label']
	training_file['result'] = training_file[['group', 'predict_group']].apply(Evaluation_matrix, axis=1)
	c_training = Counter(training_file['result'])
	training_AA = c_training["TP"]
	training_BB = c_training["FN"]
	training_CC = c_training["FP"]
	training_DD = c_training["TN"]
	training_Accuracy, training_precision, training_NPV, training_Sensitivity, training_Specificity, training_F1, training_MCC = calculate_MCC(
		training_AA, training_BB, training_CC, training_DD)
	return("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % 
		(n_estimators_option,learning_rate_option,num_leaves_option, max_depth_option,min_child_samples_option, reg_alpha_option,reg_lambda_option,AAAAA,train_MCC,test_MCC,training_MCC,AUC_cross_score))

if __name__ == '__main__':
	result_file = open('out_file', 'a')
	result_file_header = "n_estimators_option,learning_rate_option,num_leaves_option, max_depth_option,min_child_samples_option, reg_alpha_option,reg_lambda_option,AAAAA,train_MCC,test_MCC,training_MCC,AUC_cross_score\n"
	result_file.write(result_file_header)

	LGBboost_n_estimators = np.arange(100,1501,100)
	LGBboost_learning_rate =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	LGBboost_num_leaves = [63,127,255]
	LGBboost_max_depth = np.arange(3,feaure_number)
	LGBboost_min_child_samples = np.arange(10,201,10)
	LGBboost_reg_alpha = [0,0.01,0.03,0.05,0.07,0.1,0.3,0.5]
	LGBboost_reg_lambda = [0,0.01,0.03,0.05,0.07,0.1,0.3,0.5]
	

	for n_estimators_option in LGBboost_n_estimators:
		for learning_rate_option in LGBboost_learning_rate:
			for num_leaves_option in LGBboost_num_leaves:
				for max_depth_option in LGBboost_max_depth:
					for min_child_samples_option in LGBboost_min_child_samples:
						for reg_alpha_option in LGBboost_reg_alpha:
							for reg_lambda_option in LGBboost_reg_lambda:
								machine_result = machine_learning(n_estimators_option,learning_rate_option,num_leaves_option,max_depth_option,min_child_samples_option,reg_alpha_option,reg_lambda_option)
								result_file.write(machine_result + "\n")

	result_file.close()
