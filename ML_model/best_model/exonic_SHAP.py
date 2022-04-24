import numpy as np
import pandas as pd
from collections import Counter
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score,recall_score,precision_score,plot_roc_curve, f1_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.utils import class_weight
import shap
import matplotlib.pyplot as plt

training = pd.read_csv(exonic_training_file)
X_training = training[['CADD_splice','spliceAI','MMsplicing_abs','Trap','dbscSNV_ADA_SCORE','dbscSNV_RF_SCORE','SPIDEX_dpsi_abs','SPIDEX_zscore_abs','maxscant_diff_abs']]
X_training.columns = ['CADD-splice','SpliceAI','MMsplicine','TraP','dbscSNV-ADA','dbscSNV-RF','SPIDEX-dpsi','SPIDEX-zscore','MaxEntscan']
y_training = training['group']
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.25, random_state = 666, stratify=y_training)
train_classes_weights = class_weight.compute_sample_weight(class_weight='balanced',y=y_train)
machine_model = xgb.XGBClassifier(objective= 'binary:logistic', learning_rate =0.4, n_estimators=700, max_depth=3,eval_metric='logloss',subsample=0.7, colsample_bytree=0.7, gamma=2, min_child_weight=1,
                                      random_state=666, use_label_encoder=False,nthread=15,n_jobs=1)
machine_model.fit(X_train, y_train, sample_weight=train_classes_weights)

shap_values = shap.TreeExplainer(machine_model).shap_values(X_train)

plt.figure(figsize=(8, 6))
font={'family':'Arial',
      'weight':'regular',
      }
shap.summary_plot(shap_values, X_train, plot_type="bar")
plt.savefig("exonic_SHAP.pdf", bbox_inches='tight')
