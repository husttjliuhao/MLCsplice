import numpy as np
import pandas as pd
from collections import Counter
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score,recall_score,precision_score,plot_roc_curve, f1_score
from sklearn.utils import class_weight
import shap
import matplotlib.pyplot as plt
from catboost import Pool
import catboost as cb
from sklearn.utils.class_weight import compute_class_weight

training = pd.read_csv(core_donor_training_file)
X_training = training[['CADD_splice','spliceAI','MMsplicing_abs','SCAP','Trap','dbscSNV_ADA_SCORE','dbscSNV_RF_SCORE','RegSNPs','SPIDEX_dpsi_abs','SPIDEX_zscore_abs','maxscant_diff_abs']]
X_training.columns = ['CADD-splice','SpliceAI','MMsplicine','S-CAP','TraP','dbscSNV-ADA','dbscSNV-RF','RegSNPs-intron','SPIDEX-dpsi','SPIDEX-zscore','MaxEntscan']
y_training = training['group']
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.25, random_state = 666, stratify=y_training)
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
train_class_weights = dict(zip(classes, weights))
cat_features = None
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

machine_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.02, depth=9,l2_leaf_reg=7,random_strength=30,
                             bagging_temperature=0.1,border_count=254,od_wait=50,od_type='Iter',eval_metric='AUC',class_weights=train_class_weights, 
                             loss_function='Logloss',task_type='CPU',metric_period=1,cat_features=cat_features,
                             fold_len_multiplier=1.1,logging_level='Silent',random_seed=666)

machine_model.fit(train_pool, eval_set=test_pool,silent=True)

shap_values = shap.TreeExplainer(machine_model).shap_values(X_train)

plt.figure(figsize=(8, 6))
font={'family':'Arial',
      'weight':'regular',
      }
shap.summary_plot(shap_values, X_train, plot_type="bar")
plt.savefig("core_donor_SHAP.pdf", bbox_inches='tight')
