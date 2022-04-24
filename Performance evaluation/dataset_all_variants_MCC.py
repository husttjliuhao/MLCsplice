import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt

def handle_data(exonic_MCC,core_donor_MCC,extend_donor_MCC,core_acceptor_MCC,extend_acceptor_MCC,out_csv):
  df_exonic_full = pd.read_csv(exonic_MCC)
  df_exonic = df_exonic_full[['soft_name','missing','TP','FN','FP','TN']]
  
  df_core_donor_full = pd.read_csv(core_donor_MCC)
  df_core_donor = df_core_donor_full[['soft_name','missing','TP','FN','FP','TN']]
  
  df_extend_donor_full = pd.read_csv(extend_donor_MCC)
  df_extend_donor = df_extend_donor_full[['soft_name','missing','TP','FN','FP','TN']]
  
  df_core_acceptor_full = pd.read_csv(core_acceptor_MCC)
  df_core_acceptor = df_core_acceptor_full[['soft_name','missing','TP','FN','FP','TN']]
  
  df_extend_acceptor_full = pd.read_csv(extend_acceptor_MCC)
  df_extend_acceptor = df_extend_acceptor_full[['soft_name','missing','TP','FN','FP','TN']]
  
  df1 = pd.concat([df_exonic, df_core_donor, df_extend_donor, df_core_acceptor, df_extend_acceptor])
  df2 = df1.groupby(['soft_name']).agg({"missing": "sum", "TP": "sum", "FN": "sum", "FP": "sum", "TN": "sum"}).reset_index()
  df3 = df2[['soft_name','missing','TP','FN','FP','TN']]
  df3['Accuracy']= (df3['TP']+df3['TN'])/(df3['TP']+df3['FN']+df3['FP']+df3['TN'])
  df3['precision']= df3['TP']/(df3['TP']+df3['FP'])
  df3['NPV']=df3['TN']/(df3['FN']+df3['TN'])
  df3['Sensitivity']= df3['TP']/(df3['TP']+df3['FN'])
  df3['Specificity']= df3['TN']/(df3['FP']+df3['TN'])
  df3['F1']= 2*df3['precision']*df3['Sensitivity']/(df3['precision']+df3['Sensitivity'])
  df3['numerator'] = (df3['TP'] * df3['TN']) - (df3['FP'] * df3['FN'])
  df3['number'] = (df3['TP'] + df3['FP']) * (df3['TP'] + df3['FN']) * (df3['TN'] + df3['FP']) * (df3['TN'] + df3['FN'])
  df3['denominator'] = df3['number'].apply(np.sqrt)
  df3['MCC'] = df3['numerator']/df3['denominator']
  df3['all_mutation'] = df3['missing']+df3['TP']+df3['FN']+df3['FP']+df3['TN']
  df3_last = df3[['soft_name', 'missing', 'TP', 'FN', 'FP', 'TN','Accuracy', 'precision', 'NPV','Sensitivity', 'Specificity', 'F1', 'MCC', 'all_mutation']]
  df3_last.to_csv(out_csv, sep='\t', index=False)

if __name__ == '__main__':
        handle_data('exonic_MCC_file,core_donor_MCC_file,extend_donor_MCC_file,core_acceptor_MCC_file,extend_acceptor_MCC_file,out_csv_file)
