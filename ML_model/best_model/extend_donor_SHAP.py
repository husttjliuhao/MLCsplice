import numpy as np
import pandas as pd
from collections import Counter
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score,recall_score,precision_score,plot_roc_curve, f1_score
from sklearn.utils import class_weight
import shap
import matplotlib.pyplot as plt
import lightgbm
from lightgbm import LGBMClassifier


training = pd.read_csv(extend_donor_training_file)
X_training = training[['CADD_splice','spliceAI','MMsplicing_abs','Trap','RegSNPs','SPIDEX_dpsi_abs','SPIDEX_zscore_abs']]
X_training.columns = ['CADD-splice','SpliceAI','MMsplicine','TraP','RegSNPs-intron','SPIDEX-dpsi','SPIDEX-zscore']
y_training = training['group']
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.25, random_state = 666, stratify=y_training)
machine_model = lightgbm.LGBMClassifier(boosting_type='gbdt', n_estimators=500, learning_rate=0.9, num_leaves=255,max_depth=7, min_child_samples=240,reg_alpha=0.5, reg_lambda=0.07,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,objective='binary',metrics='auc',class_weight='balanced',random_state=666)
machine_model.fit(X_train, y_train)
shap_values = shap.TreeExplainer(machine_model).shap_values(X_train)

plt.figure(figsize=(8, 6))
font={'family':'Arial',
      'weight':'regular',
      }
shap.summary_plot(shap_values[0], X_train, plot_type="bar")
plt.savefig("extend_donor_SHAP.pdf", bbox_inches='tight')
