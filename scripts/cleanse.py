# Script to run a few data filtering and transformation operations
import os, datetime, random, argparse
import pandas as pd
import datetime as dt

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="filepaths")
    parser.add_argument("--input_file_path", help='Input file path')
    parser.add_argument("--output_file_path", help='Output file path')
    parser.add_argument("--filename", help='Filename')
    return parser.parse_args(argv)

def cleanse(source=None):
    df = pd.read_csv(source)
    mth_qtr_mapping = {1:3,2:6,3:9,4:12}

    # Filtering conditions
    df = df [ (df['frequency'] =='quarterly') &
            (df['level'] == 'MSA') &
            (df['hpi_type'] == 'traditional') &
            (df['hpi_flavor'] == 'all-transactions')
            ]
    '''
    As this passes to the OutputDatasetFileConfig, string specifics like commas
    in the name of the place can cause errors like pandas.errors.ParserError: Error tokenizing data. C error:
    Calling read(nbytes) on source failed. Try engine='python'. To avoid that, do
    some string formatting here to make this a tighter process.
    '''
    # String formatting
    df['place_name'] = df['place_name'].str.replace(',','|')
    df['place_name'] = df['place_name'].str.replace(' ','')
    
    # Added columns, for date changes
    df['month'] = df['quarter_period'].map(mth_qtr_mapping).astype(str)
    df = df.rename(columns={'yr':'year'})
    df['year'] = df['year'].astype(str)
    df['day'] = '1'
    df['Date']= pd.to_datetime(df[['year','month','day']])

    # Given data gaps between MSAs, starting from Q4, 2000
    df = df.loc [ df['Date'] >= '2000-11-01']

    # Drop unneeded columns
    df = df.drop(['frequency', 'level', 'hpi_type', 
        'hpi_flavor','year', 'quarter_period','day','month',
        'index_sa', 'place_id'], axis=1)

    return df

if __name__ == "__main__":
    args = getArgs()
    print(f'Input args: {args.input_file_path}')
    print(f'Output args: {args.output_file_path}')
    print(f'Filename: {args.filename}')
    df = cleanse(source=args.input_file_path)
    df.to_csv(args.output_file_path + '/' + args.filename, index=False)
