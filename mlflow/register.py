import mlflow
import pandas as pd

filter_string="tags.mlflow.project.entryPoint='main'"
df = mlflow.search_runs(0, filter_string=filter_string, order_by=["metrics.f1_weighted DESC"])
print(df)
run_id=df.loc[df['metrics.f1_weighted'].idxmax()]['run_id']

model_uri = mlflow.get_artifact_uri().rpartition("/0/")[0] + "/0/" + run_id + "/artifacts/model"
mlflow.register_model(model_uri,"Complaint classification model")

