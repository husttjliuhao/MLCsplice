import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


df_full = pd.read_csv('exon_training_all_mutation_analysis_2022.txt', sep='\t')
df_full['group'] = df_full['label'].replace(['case', 'control'], [1, 0])
df_full = df_full.fillna('NA')
df = df_full[['group','CADD_splice','spliceAI','MMsplicing_abs','Trap','dbscSNV_ADA_SCORE','dbscSNV_RF_SCORE',
              'SPIDEX_dpsi_abs','SPIDEX_zscore_abs','maxscant_diff_abs']]
df.to_csv('simple_exon_training_all_mutation_analysis_2022.txt', index=False, sep='\t')

df_CADD_splice = df[['group', 'CADD_splice']]
df_CADD_splice_last = df_CADD_splice[~df_CADD_splice['CADD_splice'].isin(['NA'])]
CADD_splice_labels = df_CADD_splice_last['group']
CADD_splice_preds = df_CADD_splice_last['CADD_splice']
CADD_splice_precision, CADD_splice_recall, CADD_splice_thresholds = precision_recall_curve(CADD_splice_labels, CADD_splice_preds)
AUPRC_CADD_splice = auc(CADD_splice_recall, CADD_splice_precision)

df_spliceAI = df[['group', 'spliceAI']]
df_spliceAI_last = df_spliceAI[~df_spliceAI['spliceAI'].isin(['NA'])]
spliceAI_labels = df_spliceAI_last['group']
spliceAI_preds = df_spliceAI_last['spliceAI']
spliceAI_precision, spliceAI_recall, spliceAI_thresholds = precision_recall_curve(spliceAI_labels, spliceAI_preds)
AUPRC_spliceAI = auc(spliceAI_recall, spliceAI_precision)

df_MMsplicing_abs = df[['group', 'MMsplicing_abs']]
df_MMsplicing_abs_last = df_MMsplicing_abs[~df_MMsplicing_abs['MMsplicing_abs'].isin(['NA'])]
MMsplicing_abs_labels = df_MMsplicing_abs_last['group']
MMsplicing_abs_preds = df_MMsplicing_abs_last['MMsplicing_abs']
MMsplicing_abs_precision, MMsplicing_abs_recall, MMsplicing_abs_thresholds = precision_recall_curve(MMsplicing_abs_labels, MMsplicing_abs_preds)
AUPRC_MMsplicing_abs = auc(MMsplicing_abs_recall, MMsplicing_abs_precision)

df_Trap = df[['group', 'Trap']]
df_Trap_last = df_Trap[~df_Trap['Trap'].isin(['NA'])]
Trap_labels = df_Trap_last['group']
Trap_preds = df_Trap_last['Trap']
Trap_precision, Trap_recall, Trap_thresholds = precision_recall_curve(Trap_labels, Trap_preds)
AUPRC_Trap = auc(Trap_recall, Trap_precision)

df_dbscSNV_ADA_SCORE = df[['group', 'dbscSNV_ADA_SCORE']]
df_dbscSNV_ADA_SCORE_last = df_dbscSNV_ADA_SCORE[~df_dbscSNV_ADA_SCORE['dbscSNV_ADA_SCORE'].isin(['NA'])]
dbscSNV_ADA_SCORE_labels = df_dbscSNV_ADA_SCORE_last['group']
dbscSNV_ADA_SCORE_preds = df_dbscSNV_ADA_SCORE_last['dbscSNV_ADA_SCORE']
dbscSNV_ADA_SCORE_precision, dbscSNV_ADA_SCORE_recall, dbscSNV_ADA_SCORE_thresholds = precision_recall_curve(dbscSNV_ADA_SCORE_labels, dbscSNV_ADA_SCORE_preds)
AUPRC_dbscSNV_ADA_SCORE = auc(dbscSNV_ADA_SCORE_recall, dbscSNV_ADA_SCORE_precision)

df_dbscSNV_RF_SCORE = df[['group', 'dbscSNV_RF_SCORE']]
df_dbscSNV_RF_SCORE_last = df_dbscSNV_RF_SCORE[~df_dbscSNV_RF_SCORE['dbscSNV_RF_SCORE'].isin(['NA'])]
dbscSNV_RF_SCORE_labels = df_dbscSNV_RF_SCORE_last['group']
dbscSNV_RF_SCORE_preds = df_dbscSNV_RF_SCORE_last['dbscSNV_RF_SCORE']
dbscSNV_RF_SCORE_precision, dbscSNV_RF_SCORE_recall, dbscSNV_RF_SCORE_thresholds = precision_recall_curve(dbscSNV_RF_SCORE_labels, dbscSNV_RF_SCORE_preds)
AUPRC_dbscSNV_RF_SCORE = auc(dbscSNV_RF_SCORE_recall, dbscSNV_RF_SCORE_precision)

df_SPIDEX_dpsi_abs = df[['group', 'SPIDEX_dpsi_abs']]
df_SPIDEX_dpsi_abs_last = df_SPIDEX_dpsi_abs[~df_SPIDEX_dpsi_abs['SPIDEX_dpsi_abs'].isin(['NA'])]
SPIDEX_dpsi_abs_labels = df_SPIDEX_dpsi_abs_last['group']
SPIDEX_dpsi_abs_preds = df_SPIDEX_dpsi_abs_last['SPIDEX_dpsi_abs']
SPIDEX_dpsi_abs_precision, SPIDEX_dpsi_abs_recall, SPIDEX_dpsi_abs_thresholds = precision_recall_curve(SPIDEX_dpsi_abs_labels, SPIDEX_dpsi_abs_preds)
AUPRC_SPIDEX_dpsi_abs = auc(SPIDEX_dpsi_abs_recall, SPIDEX_dpsi_abs_precision)

df_SPIDEX_zscore_abs = df[['group', 'SPIDEX_zscore_abs']]
df_SPIDEX_zscore_abs_last = df_SPIDEX_zscore_abs[~df_SPIDEX_zscore_abs['SPIDEX_zscore_abs'].isin(['NA'])]
SPIDEX_zscore_abs_labels = df_SPIDEX_zscore_abs_last['group']
SPIDEX_zscore_abs_preds = df_SPIDEX_zscore_abs_last['SPIDEX_zscore_abs']
SPIDEX_zscore_abs_precision, SPIDEX_zscore_abs_recall, SPIDEX_zscore_abs_thresholds = precision_recall_curve(SPIDEX_zscore_abs_labels, SPIDEX_zscore_abs_preds)
AUPRC_SPIDEX_zscore_abs = auc(SPIDEX_zscore_abs_recall, SPIDEX_zscore_abs_precision)

df_maxscant_diff_abs = df[['group', 'maxscant_diff_abs']]
df_maxscant_diff_abs_last = df_maxscant_diff_abs[~df_maxscant_diff_abs['maxscant_diff_abs'].isin(['NA'])]
maxscant_diff_abs_labels = df_maxscant_diff_abs_last['group']
maxscant_diff_abs_preds = df_maxscant_diff_abs_last['maxscant_diff_abs']
maxscant_diff_abs_precision, maxscant_diff_abs_recall, maxscant_diff_abs_thresholds = precision_recall_curve(maxscant_diff_abs_labels, maxscant_diff_abs_preds)
AUPRC_maxscant_diff_abs = auc(maxscant_diff_abs_recall, maxscant_diff_abs_precision)


lw = 3
font={'family':'Arial','weight':'540','size':'10'}
plt.figure(figsize=(6.5,6.5))
plt.plot(CADD_splice_recall, CADD_splice_precision, color='darkorange', lw=lw, linestyle='-.', label='CADD-Splice (%0.4f)' % AUPRC_CADD_splice)
plt.plot(dbscSNV_RF_SCORE_recall, dbscSNV_RF_SCORE_precision, color='magenta', lw=lw, linestyle='-', label='dbscSNV-RF (%0.4f)' % AUPRC_dbscSNV_RF_SCORE)
plt.plot(dbscSNV_ADA_SCORE_recall, dbscSNV_ADA_SCORE_precision, color='red', lw=lw, linestyle='dotted', label='dbscSNV-ADA (%0.4f)' % AUPRC_dbscSNV_ADA_SCORE)
plt.plot(Trap_recall, Trap_precision, color='gold', lw=lw, linestyle='--', label='TraP (%0.4f)' % AUPRC_Trap)
plt.plot(MMsplicing_abs_recall, MMsplicing_abs_precision, color='brown', lw=lw, linestyle='-.', label='MMSplice (%0.4f)' % AUPRC_MMsplicing_abs)
plt.plot(spliceAI_recall, spliceAI_precision, color='blue', lw=lw, linestyle='-', label='SpliceAI (%0.4f)' % AUPRC_spliceAI)
plt.plot(maxscant_diff_abs_recall, maxscant_diff_abs_precision, color='dimgrey', linestyle='dotted', lw=lw, label='MaxEntscan (%0.4f)' % AUPRC_maxscant_diff_abs)
plt.plot(SPIDEX_dpsi_abs_recall, SPIDEX_dpsi_abs_precision, color='deepskyblue', linestyle='--', lw=lw, label='SPIDEX-dpsi (%0.4f)' % AUPRC_SPIDEX_dpsi_abs)
plt.plot(SPIDEX_zscore_abs_recall, SPIDEX_zscore_abs_precision, color='steelblue', linestyle='-.', lw=lw, label='SPIDEX-zscore (%0.4f)' % AUPRC_SPIDEX_zscore_abs)

plt.xlim([0.0, 1.05])
plt.ylim([0.5, 1.03])
plt.xlabel('Recall', fontsize=10)
plt.ylabel('Precision', fontsize=10)
plt.title('exonic region', fontsize=15)
plt.legend(loc="lower right", prop=font)
plt.show()
