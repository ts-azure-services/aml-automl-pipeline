import argparse
from authentication import ws
from azureml.core.model import Model
from azureml.core import Dataset
from azureml.core.run import Run, _OfflineRun

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="filepaths")
    parser.add_argument("--model_name", help='Model name', required=True)
    parser.add_argument("--model_path", help='Model path', required=True)
    return parser.parse_args(argv)

def main():
    """Main operational flow"""
    args = getArgs()
    print(f'Model name is: {args.model_name}')
    print(f'Model path is: {args.model_path}')

    run = Run.get_context()
    prepped_data = Dataset.get_by_name(workspace=ws, name='prepped_data')

    # Get best model
    model = Model.register(
            workspace=ws, 
            model_path=args.model_path, 
            model_name=args.model_name,
            sample_input_dataset=prepped_data,
            description="Housing price AutoML time series forecasting",
            tags={'type':'time series', 'area':'housing prices'}
            )

if __name__ == "__main__":
    main()
