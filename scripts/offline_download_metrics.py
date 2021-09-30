# Script to retrieve results in offline mode
import json, pickle
from authentication import ws
import pandas as pd
from azureml.core.model import Model, Dataset
from azureml.core.run import Run#, _OfflineRun

run_id = 'f96fd978-f615-4e34-8ae5-6a0c668d10e8' # main run, under Experiment
experiment = ws.experiments['double-pipeline']
run = next(run for run in experiment.get_runs() if run.id == run_id)

automl_run = next(r for r in run.get_children() if r.name =='Time_Series_Forecasting')
outputs = automl_run.get_outputs()
#print(type(outputs)) # <class 'dict'>
#print(outputs) 
#{
#   'metrics_data': <azureml.pipeline.core.run.StepRunOutput object at 0x7f8d609d03d0>, 
#   'default_model_Time_Series_Forecasting': <azureml.pipeline.core.run.StepRunOutput object at 0x7f8d609d0790>, 
#   'best_model_data': <azureml.pipeline.core.run.StepRunOutput object at 0x7f8d609d05e0>, 
#   'default_metrics_Time_Series_Forecasting': <azureml.pipeline.core.run.StepRunOutput object at 0x7f8d70b73fd0>
#   }

metrics = outputs['default_metrics_Time_Series_Forecasting']
model = outputs['default_model_Time_Series_Forecasting']
metrics.get_port_data_reference().download('.')
model.get_port_data_reference().download('.')
