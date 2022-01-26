import pandas as pd
import json

""" This code snippet above shows the metrics file being loaded from the local location. Once you've
deserialized it and converted it to a Pandas DataFrame, you can see detailed metrics for each of the
iterations of the automated ML step."""

metrics_filename = './azureml/e32a3827-6c2f-4748-b9d1-daa0765b7009/default_metrics_Time_Series_Forecasting'
with open(metrics_filename) as f:
   metrics_output_result = f.read()
   
deserialized_metrics_output = json.loads(metrics_output_result)
df = pd.DataFrame(deserialized_metrics_output)
df = df.T
df.to_csv('transposed_metrics.csv')
print(df)
