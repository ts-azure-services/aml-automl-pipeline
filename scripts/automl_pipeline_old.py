# Decommissioned script
# Highlights breaking a pipeline into two steps and that it is possible to run it separately
# However, this will trigger independent runs which will break continuity

# Import in data, do some processing and break up the file
#import logging
from authentication import ws
from azureml.core import Dataset, ScriptRunConfig, Environment
from azureml.core.experiment import Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration, DEFAULT_CPU_IMAGE, DockerConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import Pipeline, PipelineData, TrainingOutput
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.data import OutputFileDatasetConfig
from azureml.data.data_reference import DataReference
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from azureml.automl.core.featurization.featurizationconfig import FeaturizationConfig
from azureml.pipeline.steps import AutoMLStep

### Set up resources and run configuration
compute_target = ComputeTarget(workspace=ws, name='newcluster1')
experiment = Experiment(ws, name='double-pipeline')
docker_config = DockerConfiguration(use_docker=True)
run_config = RunConfiguration()
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
run_config.docker = docker_config
run_config.environment.python.user_managed_dependencies = False
run_config.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['pandas', 'pip','python-dotenv'])
#run_config.environment.python.conda_dependencies = CondaDependencies.add_pip_package(['dotenv'])

# Pipeline step 1: Cleanup file
def_blob_store = ws.get_default_datastore()
ds = Dataset.get_by_name(workspace=ws, name='HPI_file_dataset')
intermediate_source = OutputFileDatasetConfig(destination=(def_blob_store,'/prep/')).as_mount()
intermediate_filename = 'solution.csv'
cleanup_step = PythonScriptStep(
    name="cleanup_step",
    source_directory=".",
    script_name="cleanse.py",
    compute_target=compute_target,
    arguments=[
        "--input_file_path", ds.as_named_input('starting_input').as_mount(),
        "--output_file_path", intermediate_source,
        "--filename", intermediate_filename
        ],
    runconfig=run_config,
    allow_reuse=False
    )

# Pipeline step 2: Break up file, and store results to blob
final_source = OutputFileDatasetConfig(destination=(def_blob_store,'/prep/'))#.as_mount()
final_filename = 'prepped.csv'
breakup_step = PythonScriptStep(
    name="breakup_step",
    source_directory=".",
    script_name="breakup.py",
    compute_target=compute_target,
    arguments=[
        "--input_file_path", intermediate_source.as_input(),
        "--filename", intermediate_filename,
        "--output_file_path", final_source,
        "--ffilename", final_filename
        ],
    runconfig=run_config,
    allow_reuse=False
)

# Pipeline step 3: Upload the processed dataset, and register it for further processing
processed_step = PythonScriptStep(
    name="register_dataset",
    source_directory=".",
    script_name="register_dataset_old.py",
    compute_target=compute_target,
    arguments=[
        "--input_file_path", final_source.as_input(),
        "--filename", final_filename,
        ],
    runconfig=run_config,
    allow_reuse=False
)

# Actual pipeline integration
steps = [ cleanup_step, breakup_step, processed_step ]
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion()


#prepped_data = final_source.read_delimited_files()
#print(f'PREPPED DATA: \n {prepped_data}') # should be OutputTabularDatasetConfig

### The AutoMLStep configures its dependencies automatically during job submission 
ds = Dataset.get_by_name(ws, 'prepped_data')
### Specify automated ML outputs

metrics_data = PipelineData(
        name='metrics_data',
        datastore=def_blob_store,
        pipeline_output_name='metrics_output',
        training_output=TrainingOutput(type='Metrics')
        )

model_data = PipelineData(
        name='best_model_data',
        datastore=def_blob_store,
        pipeline_output_name='model_output',
        training_output=TrainingOutput(type='Model')
        )

# Setup the forecasting parameters
forecasting_parameters = ForecastingParameters(
    time_column_name='Date',
    forecast_horizon=4,
    target_rolling_window_size=3,
    feature_lags='auto',
    validate_parameters=True
)

# Setup the classifier

## Featurization Configuration
#featurization_config = FeaturizationConfig()
#featurization_config.add_column_purpose('Date','DateTime')
#featurization_config.add_column_purpose('Place','Numeric')
##featurization_config.add_transformer_params('Imputer', ['Load'], {"strategy": "ffill"})

automl_settings = {
    "task": 'forecasting',
    #"primary_metric":'r2_score',
    "primary_metric":'normalized_root_mean_squared_error',
    "iteration_timeout_minutes": 10,
    "experiment_timeout_hours": 0.3,
    #"featurization": featurization_config,
    #"featurization": 'off',
    "compute_target":compute_target,
    "max_concurrent_iterations": 4,
    #"verbosity": logging.INFO,
    #"training_data":prepped_data,
    "training_data":ds,
    "label_column_name":'Place',
    "n_cross_validations": 5,
    #"blocked_models":['Prophet'],
    "enable_voting_ensemble":True,
    "enable_early_stopping": True,
    "model_explainability":True,
    #"enable_dnn":True,
    "forecasting_parameters": forecasting_parameters
        }

automl_config = AutoMLConfig(**automl_settings)

train_step = AutoMLStep(name='Time_Series_Forecasting',
    automl_config=automl_config,
    passthru_automl_config=False,
    outputs=[metrics_data,model_data],
    enable_default_model_output=True,
    enable_default_metrics_output=True,
    allow_reuse=False
    )

# Register the model
model_name = PipelineParameter("model_name", default_value="bestModel")
register_model_step = PythonScriptStep(
        script_name="register_model.py",
        name="register_model",
        arguments=[
            "--model_name", model_name,
            "--model_path", model_data
            ],
        inputs=[model_data],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=False
        )

# Setup experiment and trigger run
#experiment = Experiment(ws, name='automl-portion')
#pipeline = Pipeline(ws, [breakup_step, cleanup_step, train_step])
pipeline = Pipeline(ws, [train_step, register_model_step])
remote_run = experiment.submit(pipeline, show_output=True, wait_post_processing=True)
remote_run.wait_for_completion()

# Retrieve model and metrics
metrics_output_port = remote_run.get_pipeline_output('metrics_output')
model_output_port = remote_run.get_pipeline_output('model_output')
metrics_output_port.download('.', show_progress=True)
model_output_port.download('.', show_progress=True)
