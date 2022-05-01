###############################
example for the exonic region
###############################
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


df_full = pd.read_csv(predict scores for the exonic region variants file)
df_full['group'] = df_full['label'].replace(['case', 'control'], [1, 0])
df_full = df_full.fillna('NA')
df = df_full[['group','CADD_splice','spliceAI','MMsplicing_abs','Trap','dbscSNV_ADA_SCORE','dbscSNV_RF_SCORE',
              'SPIDEX_dpsi_abs','SPIDEX_zscore_abs','maxscant_diff_abs']]

df_CADD_splice = df[['group', 'CADD_splice']]
df_CADD_splice_last = df_CADD_splice[~df_CADD_splice['CADD_splice'].isin(['NA'])]
CADD_splice_labels = df_CADD_splice_last['group']
CADD_splice_preds = df_CADD_splice_last['CADD_splice']
CADD_splice_fpr, CADD_splice_tpr, CADD_splice_thresholds = roc_curve(CADD_splice_labels, CADD_splice_preds)
AUROC_CADD_splice = auc(CADD_splice_fpr, CADD_splice_tpr)

df_spliceAI = df[['group', 'spliceAI']]
df_spliceAI_last = df_spliceAI[~df_spliceAI['spliceAI'].isin(['NA'])]
spliceAI_labels = df_spliceAI_last['group']
spliceAI_preds = df_spliceAI_last['spliceAI']
spliceAI_fpr, spliceAI_tpr, spliceAI_thresholds = roc_curve(spliceAI_labels, spliceAI_preds)
AUROC_spliceAI = auc(spliceAI_fpr, spliceAI_tpr)

df_MMsplicing_abs = df[['group', 'MMsplicing_abs']]
df_MMsplicing_abs_last = df_MMsplicing_abs[~df_MMsplicing_abs['MMsplicing_abs'].isin(['NA'])]
MMsplicing_abs_labels = df_MMsplicing_abs_last['group']
MMsplicing_abs_preds = df_MMsplicing_abs_last['MMsplicing_abs']
MMsplicing_abs_fpr, MMsplicing_abs_tpr, MMsplicing_abs_thresholds = roc_curve(MMsplicing_abs_labels, MMsplicing_abs_preds)
AUROC_MMsplicing_abs = auc(MMsplicing_abs_fpr, MMsplicing_abs_tpr)

df_Trap = df[['group', 'Trap']]
df_Trap_last = df_Trap[~df_Trap['Trap'].isin(['NA'])]
Trap_labels = df_Trap_last['group']
Trap_preds = df_Trap_last['Trap']
Trap_fpr, Trap_tpr, Trap_thresholds = roc_curve(Trap_labels, Trap_preds)
AUROC_Trap = auc(Trap_fpr, Trap_tpr)

df_dbscSNV_ADA_SCORE = df[['group', 'dbscSNV_ADA_SCORE']]
df_dbscSNV_ADA_SCORE_last = df_dbscSNV_ADA_SCORE[~df_dbscSNV_ADA_SCORE['dbscSNV_ADA_SCORE'].isin(['NA'])]
dbscSNV_ADA_SCORE_labels = df_dbscSNV_ADA_SCORE_last['group']
dbscSNV_ADA_SCORE_preds = df_dbscSNV_ADA_SCORE_last['dbscSNV_ADA_SCORE']
dbscSNV_ADA_SCORE_fpr, dbscSNV_ADA_SCORE_tpr, dbscSNV_ADA_SCORE_thresholds = roc_curve(dbscSNV_ADA_SCORE_labels, dbscSNV_ADA_SCORE_preds)
AUROC_dbscSNV_ADA_SCORE = auc(dbscSNV_ADA_SCORE_fpr, dbscSNV_ADA_SCORE_tpr)

df_dbscSNV_RF_SCORE = df[['group', 'dbscSNV_RF_SCORE']]
df_dbscSNV_RF_SCORE_last = df_dbscSNV_RF_SCORE[~df_dbscSNV_RF_SCORE['dbscSNV_RF_SCORE'].isin(['NA'])]
dbscSNV_RF_SCORE_labels = df_dbscSNV_RF_SCORE_last['group']
dbscSNV_RF_SCORE_preds = df_dbscSNV_RF_SCORE_last['dbscSNV_RF_SCORE']
dbscSNV_RF_SCORE_fpr, dbscSNV_RF_SCORE_tpr, dbscSNV_RF_SCORE_thresholds = roc_curve(dbscSNV_RF_SCORE_labels, dbscSNV_RF_SCORE_preds)
AUROC_dbscSNV_RF_SCORE = auc(dbscSNV_RF_SCORE_fpr, dbscSNV_RF_SCORE_tpr)

df_SPIDEX_dpsi_abs = df[['group', 'SPIDEX_dpsi_abs']]
df_SPIDEX_dpsi_abs_last = df_SPIDEX_dpsi_abs[~df_SPIDEX_dpsi_abs['SPIDEX_dpsi_abs'].isin(['NA'])]
SPIDEX_dpsi_abs_labels = df_SPIDEX_dpsi_abs_last['group']
SPIDEX_dpsi_abs_preds = df_SPIDEX_dpsi_abs_last['SPIDEX_dpsi_abs']
SPIDEX_dpsi_abs_fpr, SPIDEX_dpsi_abs_tpr, SPIDEX_dpsi_abs_thresholds = roc_curve(SPIDEX_dpsi_abs_labels, SPIDEX_dpsi_abs_preds)
AUROC_SPIDEX_dpsi_abs = auc(SPIDEX_dpsi_abs_fpr, SPIDEX_dpsi_abs_tpr)

df_SPIDEX_zscore_abs = df[['group', 'SPIDEX_zscore_abs']]
df_SPIDEX_zscore_abs_last = df_SPIDEX_zscore_abs[~df_SPIDEX_zscore_abs['SPIDEX_zscore_abs'].isin(['NA'])]
SPIDEX_zscore_abs_labels = df_SPIDEX_zscore_abs_last['group']
SPIDEX_zscore_abs_preds = df_SPIDEX_zscore_abs_last['SPIDEX_zscore_abs']
SPIDEX_zscore_abs_fpr, SPIDEX_zscore_abs_tpr, SPIDEX_zscore_abs_thresholds = roc_curve(SPIDEX_zscore_abs_labels, SPIDEX_zscore_abs_preds)
AUROC_SPIDEX_zscore_abs = auc(SPIDEX_zscore_abs_fpr, SPIDEX_zscore_abs_tpr)

df_maxscant_diff_abs = df[['group', 'maxscant_diff_abs']]
df_maxscant_diff_abs_last = df_maxscant_diff_abs[~df_maxscant_diff_abs['maxscant_diff_abs'].isin(['NA'])]
maxscant_diff_abs_labels = df_maxscant_diff_abs_last['group']
maxscant_diff_abs_preds = df_maxscant_diff_abs_last['maxscant_diff_abs']
maxscant_diff_abs_fpr, maxscant_diff_abs_tpr, maxscant_diff_abs_thresholds = roc_curve(maxscant_diff_abs_labels, maxscant_diff_abs_preds)
AUROC_maxscant_diff_abs = auc(maxscant_diff_abs_fpr, maxscant_diff_abs_tpr)

plt.figure()
lw = 3
font={'family':'Arial','weight':'540','size':'10'}
plt.figure(figsize=(6.5,6.5))

plt.plot(CADD_splice_fpr, CADD_splice_tpr, color='darkorange', lw=lw, linestyle='-.', label='CADD-Splice (%0.4f)' % AUROC_CADD_splice)
plt.plot(dbscSNV_RF_SCORE_fpr, dbscSNV_RF_SCORE_tpr, color='magenta', lw=lw, linestyle='-', label='dbscSNV-RF (%0.4f)' % AUROC_dbscSNV_RF_SCORE)
plt.plot(dbscSNV_ADA_SCORE_fpr, dbscSNV_ADA_SCORE_tpr, color='red', lw=lw, linestyle='dotted', label='dbscSNV-ADA (%0.4f)' % AUROC_dbscSNV_ADA_SCORE)
plt.plot(Trap_fpr, Trap_tpr, color='gold', lw=lw, linestyle='--', label='TraP (%0.4f)' % AUROC_Trap)
plt.plot(MMsplicing_abs_fpr, MMsplicing_abs_tpr, color='brown', lw=lw, linestyle='-.', label='MMSplice (%0.4f)' % AUROC_MMsplicing_abs)
plt.plot(spliceAI_fpr, spliceAI_tpr, color='blue', lw=lw, linestyle='-', label='SpliceAI (%0.4f)' % AUROC_spliceAI)
plt.plot(maxscant_diff_abs_fpr, maxscant_diff_abs_tpr, color='dimgrey', linestyle='dotted', lw=lw, label='MaxEntscan (%0.4f)' % AUROC_maxscant_diff_abs)
plt.plot(SPIDEX_dpsi_abs_fpr, SPIDEX_dpsi_abs_tpr, color='deepskyblue', linestyle='--', lw=lw, label='SPIDEX-dpsi (%0.4f)' % AUROC_SPIDEX_dpsi_abs)
plt.plot(SPIDEX_zscore_abs_fpr, SPIDEX_zscore_abs_tpr, color='steelblue', linestyle='-.', lw=lw, label='SPIDEX-zscore (%0.4f)' % AUROC_SPIDEX_zscore_abs)


plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.03])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('exonic region ', fontsize=15)
plt.legend(loc="lower right", prop=font)
# plt.show()
plt.savefig('exon_region_ROC_dataset.pdf',format='pdf')

