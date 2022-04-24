import numpy as np
import pandas as pd
from collections import Counter
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score,recall_score,precision_score,plot_roc_curve, f1_score
import shap
import matplotlib.pyplot as plt
import lightgbm
from lightgbm import LGBMClassifier


training = pd.read_csv(extend_acceptor_taining_file)
X_training = training[['CADD_splice','spliceAI','MMsplicing_abs','SCAP','Trap','RegSNPs','SPIDEX_dpsi_abs','SPIDEX_zscore_abs','BPP_diff_pct_abs','LaBranchoR_diff_abs','RNABPS_diff_abs','SVMBP_diff_abs']]
X_training.columns = ['CADD-splice','SpliceAI','MMsplicine','S-CAP','TraP','RegSNPs-intron','SPIDEX-dpsi','SPIDEX-zscore','BPP','LaBranchoR','RNABPS','SVM-BPfinder']
y_training = training['group']
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.25, random_state = 666, stratify=y_training)
machine_model = lightgbm.LGBMClassifier(boosting_type='gbdt', n_estimators=400, learning_rate=0.2, num_leaves=255,max_depth=11, min_child_samples=240,reg_alpha=0.01, reg_lambda=0.01,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,objective='binary',metrics='auc',class_weight='balanced',random_state=666)
machine_model.fit(X_train, y_train)
shap_values = shap.TreeExplainer(machine_model).shap_values(X_train)
plt.figure(figsize=(8, 6))
font={'family':'Arial',
      'weight':'regular',
      }
shap.summary_plot(shap_values[1], X_train, plot_type="bar")
plt.savefig("extend_acceptor_SHAP.pdf", bbox_inches='tight')
