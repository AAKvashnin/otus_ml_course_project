import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
import pandas as pd
import logging
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def main():


  spark=(SparkSession.builder \
                        .config("spark.hadoop.yarn.timeline-service.enabled","false") \
                        .config("spark.submit.deployMode","client") \
                        .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0") \
                        .config("spark.jars","/home/alexey_kvashnin/gcs-connector-hadoop3-2.2.2.jar") \
                        .config("spark.executor.instances","4") \
                        .master("yarn") \
                        .getOrCreate())





  with mlflow.start_run() as active_run:

    mlflow.spark.autolog()
    mlflow.pyspark.ml.autolog()

    pandasDF=pd.read_csv("data/data.txt")

    sparkDF=spark.createDataFrame(pandasDF)

    model=mlflow.spark.load_model("models:/Complaint classification model/1")

    predictions=model.transform(sparkDF);

    evaluator=MulticlassClassificationEvaluator(metricName="weightedFMeasure")

    f1=evaluator.evaluate(predictions)

    mlflow.log_metric("f1_weighted",f1)

    spark.stop()




if __name__ == "__main__":
  main()










