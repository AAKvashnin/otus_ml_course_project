#!/bin/bash
gcloud auth login
export MLFLOW_TRACKING_URI=http://localhost:8000
mlflow run git@github.com:AAKvashnin/otus_ml_course_project.git -e predict -P test_data=/home/alexey_kvashnin/inputData.parquet -P result_data=/home/alexey_kvashnin/outputData.parquet