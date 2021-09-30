import argparse
import pandas as pd
from authentication import ws
from azureml.core import Dataset
#from datasets import register_dataset

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="filepaths")
    parser.add_argument("--input_file_path", help='Input file path')
    parser.add_argument("--filename", help='Input filename')
    #parser.add_argument("--output_file_path", help='Output file path')
    #parser.add_argument("--output_filename", help='Output filename')
    return parser.parse_args(argv)

def main():
    """Main operational flow"""
    args = getArgs()
    filepath = args.input_file_path + '/' + args.filename
    print(f'Filepath is: {filepath}')

    df = pd.read_csv(filepath)
    print(f' Pandas dataset:\n {df.head()}')

    ## Create Tabular Dataset
    def_blob_store = ws.get_default_datastore()
    dp = (def_blob_store, '/inter')

    # Experimental method that both creates a Tabular Dataset, and registers it as a Dataset
    fd = Dataset.Tabular.register_pandas_dataframe(df, target=dp, name='prepped_data')
    fd = fd.to_pandas_dataframe()
    print(f' Tabular dataset:\n {fd.head()}')

    ## Register the dataset
    #dataset_name = 'processed_dataset'
    #register_dataset(dataset=fd, workspace=ws, name=dataset_name)

if __name__ == "__main__":
    main()
