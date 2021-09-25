import mlflow
import mlflow.spark

model=mlflow.spark.load_model("models:/Complaint classification model/1");