#!/bin/bash
mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root hdfs:///user/alexey_kvashnin/models -h 0.0.0.0 -p 8000