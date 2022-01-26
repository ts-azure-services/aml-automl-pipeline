import os
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from scripts.authentication.service_principal import ws
from pathlib import Path
from azureml.core import Dataset
from azureml.data.dataset_factory import DataType
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def register_dataset(dataset=None, workspace=None, name=None, desc=None,tags=None):
    """Register datasets"""
    try:
        dataset = dataset.register(workspace=workspace,name=name,description=desc,tags=tags,create_new_version=True)
        print(f" Dataset registration successful for {name}")
    except Exception as e:
        print(f" Exception in registering dataset. Error is {e}")

def main():
    """Main operational flow"""
    # Set target locations and specific filename
    target_def_blob_store_path = '/blob-input-data/'
    input_filename = 'HPI_master.csv'

    # Get the default blob store
    def_blob_store = ws.get_default_datastore()

    # Upload files to blob store
    def_blob_store.upload_files(
            #files=data_file_paths, 
            files=['./input-data/HPI_master.csv'],#data_file_paths, 
            target_path=target_def_blob_store_path,
            overwrite=True,
            show_progress=True
            )
    
    # Create File Dataset
    datastore_paths = [(def_blob_store, str(target_def_blob_store_path + input_filename))]
    fd = Dataset.File.from_files(path=datastore_paths)

    # Register the dataset
    register_dataset(dataset=fd, workspace=ws, name='HPI_file_dataset')

if __name__ == "__main__":
    main()
