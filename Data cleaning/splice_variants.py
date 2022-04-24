import pandas as pd
import sqlalchemy
import multiprocessing as mp
from multiprocessing import Process
import datetime
import threading


mysql_user = ' ' # mysql user
mysql_password = ' ' # mysql code
table_name = '` `' #gtf file

create_table_sql = """
create table gtf_database.`%s`
(
    id            int                    not null AUTO_INCREMENT,
    ch_name       varchar(32) default '' not null comment 'Chr',
    pos           int         default 0  not null comment 'Pos',
    ref           varchar(256) default '' not null comment 'Ref',
    alt           varchar(256) default '' not null comment 'Alt',
    start         int         default 0  not null comment 'Start',
    end           int         default 0  not null comment 'End',
    flag          varchar(2)  default '' not null comment 'Flag',
    gene_id     varchar(32) default '' not null comment 'Gene_id',
    transcript_id varchar(32) default '' not null comment 'Transcript_id',
    gbkey         varchar(32) default '' not null comment 'mRNA',
    PRIMARY KEY (id),
    INDEX idx_start USING BTREE (start),
    INDEX idx_end USING BTREE (end),
    INDEX idx_pos USING BTREE (pos)
)
;
"""
select_sql_plus = "select * from `ncbi_all_exon_gtf` where ((start <= {pos}+50 and {pos}-2 <= start) or (end <= {pos}+2 and {pos}-13 <= end)) and ch_name='{chr_name}' and flag='+'"
select_sql_minus = "select * from `ncbi_all_exon_gtf` where ((start <= {pos}+13 and {pos}-2 <= start) or (end <= {pos}+2 and {pos}-50 <= end)) and ch_name='{chr_name}' and flag='-'"
base_insert_header = "insert into `%s` (ch_name, pos, ref, alt, exon, start, end, flag, gene_id, transcript_id, gbkey, exon_number ) values "
insert_header = ""
insert_value = "('%s', %d, '%s', '%s', %d, %d, '%s', '%s', '%s', '%s')"
engine = sqlalchemy.create_engine("mysql+pymysql://{}:{}@ localhost/gtf_database ".format(mysql_user, mysql_password), pool_size=20, max_overflow=5)

def create_table(result_table_name):
    sql = create_table_sql % result_table_name
    with engine.connect() as conn:
        conn.execute(sql)

def read_origin_data(origin_file_path):
    origin_df = pd.read_csv(filename, sep='\t', index_col=False)
    return origin_df

def select_insert(df_row):
    # df_row = df_row[1]
    plus_sql = select_sql_plus.format(pos=df_row['POS'], chr_name=df_row['CHROM'])
    minus_sql = select_sql_minus.format(pos=df_row['POS'], chr_name=df_row['CHROM'])
    with engine.connect() as conn:
        rs_list = []
        plus_rs = conn.execute(plus_sql)
        minus_rs = conn.execute(minus_sql)
        if plus_rs:
            rs_list.append(plus_rs)
        if minus_rs:
            rs_list.append(minus_rs)
        if not rs_list:
            return false
  
        insert_values = []
        
        for rs in rs_list:
            for row in rs:
                insert_values.append(insert_value % (row[1], df_row['POS'], df_row['REF'], df_row['ALT'], row[2], row[3], row[4], row[5], row[6], row[7]))
        if insert_values:
            insert_sql = insert_header + ','.join(insert_values)
            print(insert_sql)
            conn.execute(insert_sql)
            return True
        else:
            return False
          
def export_data(result_table_name):
    df_sql = 'select * from `%s`' % result_table_name
    print(df_sql)
    with engine.connect() as conn:
        df = pd.read_sql(df_sql, conn)
        df.to_csv(result_table_name + '.csv', sep='\t', index=False)

def threading_target(df):
    df.apply(select_insert, axis=1)

def origin_file_path(filename):
    result_table_name = 'resut_'+filename
    create_table(result_table_name)
    global base_insert_header, insert_header
    insert_header = base_insert_header % result_table_name
    df = read_origin_data(filename)
    row_count = df.shape[0]
    echo_count = row_count // 4
    s_col = 0
    e_col = 0

    threads = []
    for i in range(4):
        s_col = i * echo_count
        e_col = (i+1) * echo_count
        print(s_col, e_col)
        target_df = df[s_col:e_col]
        t = threading.Thread(target=threading_target, args=(target_df, ))
        threads.append(t)
        t.start()

    if e_col < row_count:
        target_df = df[e_col:row_count]
        print(e_col, row_count)
        t = threading.Thread(target=threading_target, args=(target_df, ))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    export_data(result_table_name)
    print("----end-----")
   
if __name__ == '__main__':
        origin_file_path(filename)
