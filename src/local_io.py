import os, sqlite3
import pandas as pd

from constants import min_peptide_len, max_peptide_len
from tensorize import modseq_to_codedseq


def read_rt_database( db_loc ):
    df = pd.read_csv( db_loc, delimiter='\t', )
    df['CodedPeptideSeq'] = [ modseq_to_codedseq( p ) for p in df.PeptideModSeq ]
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


def read_fasta( file_name ): # Return dict with keys = names, values = seqs
    seq_dict = {}
    trigger = 0
    for line in open( file_name, 'r' ):
        if line[0] == '>': # New seq
            if trigger != 0:
                seq_dict[ name ] = seq
            name = line[1:].rstrip()
            seq = ''
            trigger = 1
        else:
            seq += line.rstrip()
    seq_dict[name] = seq # Add the final sequence since 
    return seq_dict
    
    

dlib_dtypes= { 'metadata' :        { 'Key' :                             'string',
                                     'Value' :                           'string',
                                   },
                      
               'entries' :         { 'PrecursorMz' :                     'double', 
                                     'PrecursorCharge' :                 'int', 
                                     'PeptideModSeq' :                   'string',
                                     'PeptideSeq' :                      'string',
                                     'Copies' :                          'int',
                                     'RTInSeconds' :                     'double',
                                     'Score':                            'double',
                                     'MassEncodedLength' :               'int',
                                     'MassArray' :                       'blob',
                                     'IntensityEncodedLength' :          'int',
                                     'IntensityArray' :                  'blob',
                                     'CorrelationEncodedLength' :        'int',
                                     'CorrelationArray' :                'blob',
                                     'RTInSecondsStart' :                'double',
                                     'RTInSecondsStop' :                 'double',
                                     'MedianChromatogramEncodedLength' : 'int',
                                     'MedianChromatogramArray' :         'blob',
                                     'SourceFile' :                      'string', 
                                   },
                       
              'peptidetoprotein' : { 'PeptideSeq' :                      'string',
                                     'isDecoy' :                         'boolean',
                                     'ProteinAccession' :                'string', 
                                   },
             }

def create_dlib( file_name, overwrite=True, ):
    if os.path.isfile( file_name ):
        if overwrite: 
            os.remove( file_name )
        else:
            assert False, 'DLIB already exists and overwrite set to False'
    con = sqlite3.connect( file_name )
    cursor = con.cursor()
    for table in dlib_dtypes:
        creation_string = 'CREATE TABLE ' + table + '( '
        for column in dlib_dtypes[ table ]:
            dtype_str = column + ' ' + dlib_dtypes[table][column] + ', '
            creation_string += dtype_str
        creation_string = creation_string[:-2] + ' )'
        cursor.execute( creation_string )
    cursor.execute( "INSERT INTO metadata (Key, Value) VALUES ('version', '0.1.14')" )
    con.commit()
    con.close()
    return 0

    
    
def append_table_to_dlib( table, table_name, dlib_file ):
    # Add any missing columns
    #for column in dlib_dtypes[ table_name ]:
    #    if column not in table.columns:
    #        table[ column ] = None
    
    # Append to existing dlib
    con = sqlite3.connect(dlib_file)
    cursor = con.cursor()
    table.to_sql(table_name, con, if_exists='append', index=False, )
    con.commit()
    con.close()
    return 0
