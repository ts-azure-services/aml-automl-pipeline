import pickle

model_filename = './azureml/77a28722-fcae-4e2c-ab99-9f5a289eb2dd/default_model_Time_Series_Forecasting'
with open(model_filename, "rb" ) as f:
    best_model = pickle.load(f)

print( type(best_model))
