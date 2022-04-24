##################################################################################################################
example for Rank and KNN used in core_donor_region of four test datasets
##################################################################################################################
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
def Rank_rank(test_file):
    sample_df = pd.read_csv(training_file_on_missing)
    predicted_df_full = pd.read_csv(test_file, sep='\t')
    result_df = pd.DataFrame(columns=predicted_df.columns)
    for index, row in predicted_df.iterrows():
        df_KNN = sample_df.append(row, ignore_index=True)
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        df_KNN.iloc[:, 1:12] = imputer.fit_transform(df_KNN.iloc[:, 1:12])
        df = df_KNN
        total = 3070

        df['CADD_splice_rank'] = df.CADD_splice.rank(method='min',ascending=False)
        df['CADD_splice_rank_score']= np.log10(df['CADD_splice_rank']/total)*(-10)

        df['spliceAI_rank'] = df.spliceAI.rank(method='min',ascending=False)
        df['spliceAI_rank_score']= np.log10(df['spliceAI_rank']/total)*(-10)

        df['MMsplicing_abs_rank'] = df.MMsplicing_abs.rank(method='min',ascending=False)
        df['MMsplicing_abs_rank_score']= np.log10(df['MMsplicing_abs_rank']/total)*(-10)

        df['SCAP_rank'] = df.SCAP.rank(method='min',ascending=False)
        df['SCAP_rank_score']= np.log10(df['SCAP_rank']/total)*(-10)

        df['Trap_rank'] = df.Trap.rank(method='min',ascending=False)
        df['Trap_rank_score']= np.log10(df['Trap_rank']/total)*(-10)

        df['dbscSNV_ADA_SCORE_rank'] = df.dbscSNV_ADA_SCORE.rank(method='min',ascending=False)
        df['dbscSNV_ADA_SCORE_rank_score']= np.log10(df['dbscSNV_ADA_SCORE_rank']/total)*(-10)

        df['dbscSNV_RF_SCORE_rank'] = df.dbscSNV_RF_SCORE.rank(method='min',ascending=False)
        df['dbscSNV_RF_SCORE_rank_score']= np.log10(df['dbscSNV_RF_SCORE_rank']/total)*(-10)

        df['RegSNPs_rank'] = df.RegSNPs.rank(method='min',ascending=False)
        df['RegSNPs_rank_score']= np.log10(df['RegSNPs_rank']/total)*(-10)

        df['SPIDEX_dpsi_abs_rank'] = df.SPIDEX_dpsi_abs.rank(method='min',ascending=False)
        df['SPIDEX_dpsi_abs_rank_score']= np.log10(df['SPIDEX_dpsi_abs_rank']/total)*(-10)

        df['SPIDEX_zscore_abs_rank'] = df.SPIDEX_zscore_abs.rank(method='min',ascending=False)
        df['SPIDEX_zscore_abs_rank_score']= np.log10(df['SPIDEX_zscore_abs_rank']/total)*(-10)

        df['maxscant_diff_abs_rank'] = df.maxscant_diff_abs.rank(method='min',ascending=False)
        df['maxscant_diff_abs_rank_score']= np.log10(df['maxscant_diff_abs_rank']/total)*(-10)

        df1 = df
        df2=df1[['ID','CADD_splice_rank_score','spliceAI_rank_score','MMsplicing_abs_rank_score','SCAP_rank_score','Trap_rank_score','dbscSNV_ADA_SCORE_rank_score',
                 'dbscSNV_RF_SCORE_rank_score','RegSNPs_rank_score','SPIDEX_dpsi_abs_rank_score','SPIDEX_zscore_abs_rank_score','maxscant_diff_abs_rank_score']]

        df2.columns = ['ID','CADD_splice','spliceAI','MMsplicing_abs','SCAP','Trap','dbscSNV_ADA_SCORE','dbscSNV_RF_SCORE','RegSNPs',
                       'SPIDEX_dpsi_abs','SPIDEX_zscore_abs','maxscant_diff_abs']
                   
        df_rank_score = df2 
        result_df = result_df.append(df_rank_score.iloc[-1], ignore_index=True)
        print(index)
        print(df_rank_score.iloc[-1]['ID'])
        result_df.to_csv('Rank_KNN_' + csv, index=False, sep='\t')

if __name__ == '__main__':
    for file_name in ['MFASS_score_core_donor.txt',
                      'Vex-seq_score_core_donor.txt',
                      'literature_score_core_donor.txt',
                      'Clinvar_score_core_donor.txt']:
        Rank_rank(file_name)
