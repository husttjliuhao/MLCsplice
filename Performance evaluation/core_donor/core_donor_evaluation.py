import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt
import os

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

def handel(column_name, threshold_option):
    df_score = pd.read_csv(dataset_score_file, sep='\t')
    df_score['group'] = df_score['label'].replace(['case', 'control'], [1, 0])
    df_score = df_score.fillna('NA')
    df1 = df_score.loc[:, ['group', column_name]]
    df2 = df1[~df1[column_name].isin(['NA'])]
    df_last = df2
    number_NA = len(df_score) - len(df_last)
    y_preds = df_last[column_name]
    y_labels = df_last['group']
    df_last['predict_group'] = df_last[column_name].map(lambda x: 1 if x >= threshold_option else 0)
    df_last['result'] = df_last[['group', 'predict_group']].apply(Evaluation_matrix, axis=1)
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
    return "%s,%d,%f,%f,%f,%f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % \
           (column_name,number_NA,AA,BB,CC,DD,threshold_option,Accuracy,precision,NPV,Sensitivity,Specificity,F1,MCC)


if __name__ == '__main__':
    file_names = ['core_donor_training_optimal_th.txt']
    result_header = "soft_name,missing,TP,FN,FP,TN,optimal_th,Accuracy,precision,NPV,Sensitivity,Specificity,F1,MCC\n"
    for name in file_names:
        f = open(name, 'r')
        result_file = open('MCC_'+name, 'a')
        result_file.write(result_header)
        for line in f.readlines():
            if 'soft_name' in line:
                continue
            soft_name, opt = line.split('\t')
            opt = float(opt)
            result = handel(soft_name, opt)
            result_file.write(result + "\n")
        result_file.close()
        f.close()
