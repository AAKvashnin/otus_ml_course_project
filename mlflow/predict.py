import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
import logging
import argparse
from pyspark.sql.functions import regexp_replace,col



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Spark example")
    parser.add_argument(
        "--test-data",
        type=string,
        help="Param for test data path",
    )
    parser.add_argument(
        "--result-data",
        type=string,
        help="Param for result data path",
    )
    return parser.parse_args()



def main():

  args = parse_args()



  spark=(SparkSession.builder \
                        .config("spark.hadoop.yarn.timeline-service.enabled","false") \
                        .config("spark.submit.deployMode","client") \
                        .config("spark.executor.instances","10") \
                        .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0") \
                        .config("spark.jars","/home/alexey_kvashnin/gcs-connector-hadoop3-2.2.2.jar") \
                        .master("yarn") \
                        .getOrCreate())




  with mlflow.start_run() as active_run:

    mlflow.spark.autolog()
    mlflow.pyspark.ml.autolog()

    try:

              data = spark.read.load(args.test_data)
              cleansed_data = data.withColumn("text_cleaned",regexp_replace(col("text"),"[^a-zA-Z0-9]+", " "))
    except Exception as e:
              logger.exception("Unable to read training & test data. Error: %s", e)


    model=mlflow.spark.load_model("models:/Complaint classification model/1")

    predictions=model.transform(data).select(col("text"),col("Product_out").alias("Product"))

    predictions.write.save(args.result_data)

    spark.stop()




if __name__ == "__main__":
  main()










