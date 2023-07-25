# Databricks notebook source
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
# See this https://github.com/mlflow/mlflow/blob/13f355577cf925afefae1cd246d47a8a3517ffe1/mlflow/exceptions.py#L29C8-L29C8 for supported exceptions

# Your ExtendedModel class remains unchanged
class ExtendedModel(mlflow.pyfunc.PythonModel):
  def predict(self, context, model_input):
    if not ((model_input >= 0) & (model_input <= 100)): # Assuming model_input is a pandas DataFrame
        raise MlflowException("Input values should be between 0 and 100 (inclusive).")
    return model_input * 2

# COMMAND ----------

# Log the model
with mlflow.start_run() as run:
    model_path = "ExtendedModel"
    mlflow.pyfunc.log_model(artifact_path=model_path, python_model=ExtendedModel())

# COMMAND ----------

# Get the current run ID
run_id = run.info.run_id

# Load the model using its run ID and artifact path
loaded_model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/ExtendedModel")

# COMMAND ----------

# This should work
result = loaded_model.predict(10)
print(result)

# COMMAND ----------

# This should throw an exception
result = loaded_model.predict(-1)
print(result) 
