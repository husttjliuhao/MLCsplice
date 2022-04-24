import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_full = pd.read_csv(exonic/core_donor/extend_donor/core_acceptor/extend_acceptor_training_file)
pearsoncorr = df.corr(method='spearman')
plt.figure(figsize=(8, 6))
hmap=sns.heatmap(pearsoncorr,vmin=0, vmax=1,xticklabels=pearsoncorr.columns,
                yticklabels=pearsoncorr.columns,cmap='RdBu_r',cbar_kws = {'pad':0.05,},annot=None,fmt ='.2',linewidth=0.1)
font={'family':'Arial','weight':'regular'}
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
label_y = hmap.get_yticklabels()
plt.setp(label_y, horizontalalignment='right', weight="regular")
label_x = hmap.get_xticklabels()
plt.setp(label_x, rotation=45,horizontalalignment='right', weight="regular" )
plt.show()
plt.savefig('splice_region_spearman.pdf', format='pdf')
