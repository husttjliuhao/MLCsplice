##########################################################
example for the cutoff in extended acceptor region
##########################################################
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,roc_auc_score,plot_roc_curve
import pandas as pd
from collections import Counter
from math import sqrt

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
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
df = pd.read_csv(file_name)
df['group'] = df['label'].replace(['case', 'control'], [1, 0])
df1 = df.fillna('NA')

column_name_option = ['CADD_splice','spliceAI','MMsplicing_abs','Trap','SPIDEX_dpsi_abs','SPIDEX_zscore_abs','SCAP',
              'RegSNPs','BPP_diff_pct_abs','LaBranchoR_diff_abs','RNABPS_diff_abs','SVMBP_diff_abs']

for column_name in column_name_option:
    df2 = df1.loc[:,['group',column_name]]
    df3 = df2[~df2[column_name].isin(['NA'])]
    df_last =df3
    number_NA = len(df) - len(df_last)
    y_preds = df_last[column_name]
    y_labels = df_last['group']
    fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(y_labels, y_preds)
    df_last['predict_group'] = df_last[column_name].map(lambda x: 1 if x >= optimal_th else 0)
    df_last['result'] = df_last.loc[:,['group', 'predict_group']].apply(Evaluation_matrix, axis=1)
    c = Counter(df_last['result'])
    AA = c["TP"]
    BB = c["FN"]
    CC = c["FP"]
    DD = c["TN"]
    Accuracy=(AA+DD)/(AA+BB+CC+DD)
    precision=AA/(AA+CC)
    NPV=DD/(BB+DD)
    Sensitivity=AA/(AA+BB)
    Specificity=DD/(CC+DD)
    F1=2*precision*Sensitivity/(precision+Sensitivity)
    numerator = (AA * DD) - (CC * BB) 
    denominator = sqrt((AA + CC) * (AA + BB) * (DD + CC) * (DD + BB))
    MCC = numerator/denominator
    print("%s, %d, %.2f, %.2f, %.2f, %.2f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" %
          (column_name, number_NA, AA, BB, CC, DD, roc_auc, optimal_th, Accuracy, precision, NPV, Sensitivity, Specificity, F1, MCC))
