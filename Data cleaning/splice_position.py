import pandas as pd


def splice_position(file_name):
    df = pd.read_table(file_name, sep='\t')

    df['exonic_donor'] = 'exonic_donor'
    df['exonic_accept'] = 'exonic_accept'
    df['core_donor'] = 'core_donor'
    df['extend_donor'] = 'extend_donor'
    df['core_acceptor'] = 'core_acceptor'
    df['extend_acceptor'] = 'extend_acceptor'
    df['classical'] = 'classical'
    
    df['pos_start'] = df['pos'] - df['start']
    df['pos_end'] = df['pos'] - df['end']
    df['end-2'] = df['end'] - 2
    df['end+13'] = df['end'] + 13
    df['start-50'] = df['start'] - 50
    df['start+2'] = df['start'] + 2
    df['start-13'] = df['start'] - 13
    df['end+50'] = df['end'] + 50

    df1 = df[(df['pos'] >= df['end-2']) & (df['pos'] <= df['end+13']) & (df['flag'] == '+')]
    df_forward_exonic_donor = df1[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'exonic_donor', 'pos_end']]
    df_forward_exonic_donor['pos_end_last'] = df_forward_exonic_donor['pos_end']-1
    df_forward_exonic_donor_last = df_forward_exonic_donor[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'exonic_donor', 'pos_end_last']]
    df_forward_donor_classical = df1[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'classical', 'pos_end']]
    df_forward_core_donor = df1[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'core_donor', 'pos_end']]
    df_forward_extend_donor = df1[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'extend_donor', 'pos_end']]
    df_forward_exonic_donor_last = df_forward_exonic_donor_last[df_forward_exonic_donor_last['pos_end_last'].isin([-1, -2, -3])]
    df_forward_donor_classical = df_forward_donor_classical[df_forward_donor_classical['pos_end'].isin([1, 2])]
    df_forward_core_donor = df_forward_core_donor[df_forward_core_donor['pos_end'].isin([3, 4, 5, 6])]
    df_forward_extend_donor = df_forward_extend_donor[df_forward_extend_donor['pos_end'].isin([7, 8, 9, 10, 11, 12, 13])]

    df2 = df[(df['pos'] >= df['start-50']) & (df['pos'] <= df['start+2']) & (df['flag'] == '+')]
    df_forward_accept_exonic = df2[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'exonic_accept', 'pos_start']]
    df_forward_accept_exonic['pos_start_last'] = df_forward_accept_exonic['pos_start']+1
    df_forward_accept_exonic_last = df_forward_accept_exonic[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'exonic_accept', 'pos_start_last']]
    df_forward_accept_classical = df2[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'classical', 'pos_start']]
    df_forward_core_acceptor = df2[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'core_acceptor', 'pos_start']]
    df_forward_extend_acceptor = df2[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'extend_acceptor', 'pos_start']]
    df_forward_accept_exonic_last = df_forward_accept_exonic_last[df_forward_accept_exonic_last['pos_start_last'].isin([1, 2, 3])]
    df_forward_accept_classical = df_forward_accept_classical[df_forward_accept_classical['pos_start'].isin([-1, -2])]
    df_forward_core_acceptor = df_forward_core_acceptor[df_forward_core_acceptor['pos_start'].between(-12,-3)]
    df_forward_extend_acceptor = df_forward_extend_acceptor[df_forward_extend_acceptor['pos_start'].between(-50,-13)]

    df3 = df[(df['pos'] >= df['start-13']) & (df['pos'] <= df['start+2']) & (df['flag'] == '-')]
    df_reverse_exonic_donor = df3[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'exonic_donor', 'pos_start']]
    df_reverse_exonic_donor['pos_start_last'] = (df_reverse_exonic_donor['pos_start']+1)* (-1)
    df_reverse_exonic_donor_last = df_reverse_exonic_donor[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'exonic_donor', 'pos_start_last']]
    df_reverse_donor_classical = df3[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'classical', 'pos_start']]
    df_reverse_core_donor = df3[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'core_donor', 'pos_start']]
    df_reverse_core_donor['pos_start_last'] = df_reverse_core_donor['pos_start'] * (-1)
    df_reverse_core_donor_last = df_reverse_core_donor[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'core_donor', 'pos_start_last']]
    df_reverse_extend_donor = df3[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'extend_donor', 'pos_start']]
    df_reverse_extend_donor['pos_start_last'] = df_reverse_extend_donor['pos_start'] * (-1)
    df_reverse_extend_donor_last = df_reverse_extend_donor[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'extend_donor', 'pos_start_last']]
    df_reverse_exonic_donor_last = df_reverse_exonic_donor_last[df_reverse_exonic_donor_last['pos_start_last'].isin([-1, -2, -3])]
    df_reverse_donor_classical = df_reverse_donor_classical[df_reverse_donor_classical['pos_start'].isin([-1, -2])]
    df_reverse_core_donor_last = df_reverse_core_donor_last[df_reverse_core_donor_last['pos_start_last'].isin([3, 4, 5, 6])]
    df_reverse_extend_donor_last = df_reverse_extend_donor_last[df_reverse_extend_donor_last['pos_start_last'].isin([7, 8, 9, 10, 11, 12, 13])]

    df4 = df[(df['pos'] >= df['end-2']) & (df['pos'] <= df['end+50']) & (df['flag'] == '-')]
    df_reverse_accept_exonic = df4[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'exonic_accept', 'pos_end']]
    df_reverse_accept_exonic['pos_end_last'] = (df_reverse_accept_exonic['pos_end'] -1)* (-1)
    df_reverse_accept_exonic_last = df_reverse_accept_exonic[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'exonic_accept', 'pos_end_last']]
    df_reverse_accept_classical = df4[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'classical', 'pos_end']]
    df_reverse_core_acceptor = df4[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'core_acceptor', 'pos_end']]
    df_reverse_core_acceptor['pos_end_last'] = df_reverse_core_acceptor['pos_end'] * (-1)
    df_reverse_core_acceptor_last = df_reverse_core_acceptor[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'core_acceptor', 'pos_end_last']]
    df_reverse_extend_acceptor = df4[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'extend_acceptor', 'pos_end']]
    df_reverse_extend_acceptor['pos_end_last'] =  df_reverse_extend_acceptor['pos_end'] * (-1)
    df_reverse_extend_acceptor_last = df_reverse_extend_acceptor[['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'extend_acceptor', 'pos_end_last']]
    df_reverse_accept_exonic_last = df_reverse_accept_exonic_last[df_reverse_accept_exonic_last['pos_end_last'].isin([1, 2, 3])]
    df_reverse_accept_classical = df_reverse_accept_classical[df_reverse_accept_classical['pos_end'].isin([1, 2])]
    df_reverse_core_acceptor_last = df_reverse_core_acceptor_last[df_reverse_core_acceptor_last['pos_end_last'].between(-12,-3)]
    df_reverse_extend_acceptor_last = df_reverse_extend_acceptor_last[df_reverse_extend_acceptor_last['pos_end_last'].between(-50,-13)]

    df_forward_exonic_donor_last.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_forward_donor_classical.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_forward_core_donor.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_forward_extend_donor.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_forward_accept_exonic_last.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_forward_accept_classical.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_forward_core_acceptor.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_forward_extend_acceptor.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_reverse_exonic_donor_last.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_reverse_donor_classical.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_reverse_core_donor_last.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_reverse_extend_donor_last.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_reverse_accept_exonic_last.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_reverse_accept_classical.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_reverse_core_acceptor_last.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']
    df_reverse_extend_acceptor_last.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'donor_accept','pos_start_end']

    df_concat_exonic = df_forward_exonic_donor_last.append([df_forward_accept_exonic_last, df_reverse_exonic_donor_last, df_reverse_accept_exonic_last])
    df_concat_classical = df_forward_donor_classical.append([df_forward_accept_classical, df_reverse_donor_classical, df_reverse_accept_classical])
    df_concat_core_donor = df_forward_core_donor.append([df_reverse_core_donor_last])
    df_concat_extend_donor = df_forward_extend_donor.append([df_reverse_extend_donor_last])
    df_concat_core_acceptor = df_forward_core_acceptor.append([df_reverse_core_acceptor_last])
    df_concat_extend_acceptor = df_forward_extend_acceptor.append([df_reverse_extend_acceptor_last])
    df_concat_ALL = df_concat_exonic.append([df_concat_classical, df_concat_core_donor, df_concat_extend_donor, df_concat_core_acceptor, df_concat_extend_acceptor])
    df_concat_classical1 = df_concat_classical
    df_concat_classical1.columns = ['ch_name', 'pos', 'ref', 'alt', 'start', 'end', 'flag', 'classical_label','pos_start_end']
    df_concat_classical2 = df_concat_classical[['ch_name', 'pos', 'ref', 'alt', 'classical_label']]
    df_merge_nonclassical_classical = pd.merge(df_concat_ALL, df_concat_classical2, how='left',
                                               left_on=['ch_name', 'pos', 'ref', 'alt'],
                                               right_on=['ch_name', 'pos', 'ref', 'alt'])

    df_merge_nonclassical = df_merge_nonclassical_classical[~df_merge_nonclassical_classical['classical_label'].isin(['classical'])]
    df_merge_nonclassical_1 = df_merge_nonclassical[['ch_name', 'pos', 'ref', 'alt']]
    df_merge_nonclassical_last = df_merge_nonclassical_1.drop_duplicates()
    df_merge_nonclassical_classical.to_csv(nonclassical_classical_file)
    df_merge_nonclassical_last.to_csv(nonclassical_file)


if __name__ == '__main__':
    for file_name in [splice_variants_file]:
        splice_position(file_name)
