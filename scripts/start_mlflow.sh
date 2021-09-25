#!/bin/bash
gcloud auth login
mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root gs://course_project_mlflow_artifacts/mlflow_artifacts -h 0.0.0.0 -p 8000