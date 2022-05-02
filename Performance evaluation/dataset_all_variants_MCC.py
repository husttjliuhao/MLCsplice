
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, plot_roc_curve
import pandas as pd
from collections import Counter
from math import sqrt
import os


df = pd.read_csv(file_name)
df['group'] = df['label'].replace(['case', 'control'], [1, 0])
df = df.fillna('NA')

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
    df2 = df.loc[:, ['group', column_name]]
    df3 = df2[~df2[column_name].isin(['NA'])]
    df_last = df3
    number_NA = len(df) - len(df_last)
    df_last['predict_group'] = df_last[column_name].map(lambda x: 1 if x >= threshold_option else 0)
    df_last['result'] = df_last[['group', 'predict_group']].apply(Evaluation_matrix, axis=1)
    c = Counter(df_last['result'])
    AA = c["TP"]
    BB = c["FN"]
    CC = c["FP"]
    DD = c["TN"]
    try:
        Accuracy = (AA + DD) / (AA + BB + CC + DD)
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
        Sensitivity = float('nan')
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
        MCC = numerator / denominator
    except ZeroDivisionError:
        MCC = float('nan')
    return "%s,%d,%f,%f,%f,%f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % \
           (column_name, number_NA, AA, BB, CC, DD, threshold_option, Accuracy, precision, NPV,Sensitivity,Specificity,F1, MCC)


if __name__ == '__main__':
    # file_names = [name for name in os.listdir() if name.endswith('txt')]
    file_names = [cutoff_file]
    result_header = "soft_name,missing,TP,FN,FP,TN,optimal_th,Accuracy, precision, NPV,Sensitivity,Specificity,F1, MCC\n"
    for name in file_names:
        f = open(name, 'r')
        result_file = open(out_csv, 'a')  
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

