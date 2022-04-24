import numpy as np
import pandas as pd
import catboost as cb
from collections import Counter
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc,roc_auc_score,recall_score,precision_score,plot_roc_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight
from catboost import Pool

training = pd.read_csv(training_file)
array_training = training.values
X_training = array_training[:,1:feaure_number]
y_training = array_training[:,0]
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.25, random_state = 666, stratify=y_training)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
train_class_weights = dict(zip(classes, weights))
cat_features = None
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)


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

def machine_learning(iterations_option,learning_rate_option,depth_option,l2_leaf_reg_option,random_strength_option,bagging_temperature_option):
	machine_model = cb.CatBoostClassifier(iterations=iterations_option, learning_rate=learning_rate_option, depth=depth_option,l2_leaf_reg=l2_leaf_reg_option,random_strength=random_strength_option,
		                                  bagging_temperature=bagging_temperature_option,border_count=254,od_wait=50,eval_metric='AUC',class_weights=train_class_weights, 
		                                  loss_function='Logloss',task_type='CPU',metric_period=1,cat_features=cat_features,fold_len_multiplier=1.1random_seed=666)
	machine_model.fit(train_pool, eval_set=test_pool,silent=True)
	
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
	return("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % 
		(iterations_option,learning_rate_option,depth_option,l2_leaf_reg_option,random_strength_option,bagging_temperature_option,AAAAA,train_MCC,test_MCC,training_MCC,AUC_cross_score))

if __name__ == '__main__':
	result_file = open('out_file', 'a')
	result_file_header = "iterations_option,learning_rate_option,depth_option,l2_leaf_reg_option,random_strength_option,bagging_temperature_option,AAAAA,train_MCC,test_MCC,training_MCC,AUC_cross_score\n"
	result_file.write(result_file_header)

	CGBboost_iterations = np.arange(300,1501,100)
	CGBboost_learning_rate = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
	CGBboost_depth = np.arange(3,feaure_number)
	CGBboost_l2_leaf_reg = [1,3,5,7,9,11,13,15,17,19]
	CGBboost_random_strength = [10,20,30,40,50,60,70,80,90,100]
	CGBboost_bagging_temperature = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	
	for iterations_option in CGBboost_iterations:
		for learning_rate_option in CGBboost_learning_rate:
			for depth_option in CGBboost_depth:
				for l2_leaf_reg_option in CGBboost_l2_leaf_reg:
					for random_strength_option in CGBboost_random_strength:
						for bagging_temperature_option in CGBboost_bagging_temperature:
							machine_result = machine_learning(iterations_option,learning_rate_option,depth_option,l2_leaf_reg_option,random_strength_option,bagging_temperature_option)
							result_file.write(machine_result + "\n")

	result_file.close()
