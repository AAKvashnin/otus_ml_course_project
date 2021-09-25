#!/bin/bash
gcloud auth login
export MLFLOW_TRACKING_URI=http://localhost:8000
mlflow models serve -m "models:/Complaint classification model/1"