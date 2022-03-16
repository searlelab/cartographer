import sqlite3
import pandas as pd

from constants import min_peptide_len, max_peptide_len
from tensorize import modseq_to_coded_seq


def read_rt_database( db_loc ):
    df = pd.read_csv( db_loc, delimiter='\t', )
    df['CodedPeptideSeq'] = [ modseq_to_coded_seq( p ) for p in df.PeptideModSeq ]
    df['PeptideLength'] = [ len(p)-2 for p in df.CodedPeptideSeq ]
    df = df[ ( df.PeptideLength.between( min_peptide_len, max_peptide_len, ) ) &
             ( [ p.count('[') == 0 for p in df.CodedPeptideSeq ] ) ]
    return df


def read_table( sqlite_file, table_name ):
    con = sqlite3.connect( sqlite_file )
    c = con.cursor()
    df = pd.read_sql_query("SELECT * from " + table_name, con)
    con.close()
    return df



