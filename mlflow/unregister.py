import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.delete_registered_model("Complaint classification model")