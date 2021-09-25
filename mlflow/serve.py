import mlflow
import mlflow.spark
import argparse
from pyspark.sql import SparkSession


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Spark example")
    parser.add_argument(
        "--test-data",
        type=path,
       ,help="Path to test data"
    )

    return parser.parse_args()


def main():

  args = parse_args()

  model=mlflow.spark.load_model("models:/Complaint classification model/1")

  try:

          data = spark.read.load(args.test_data)
          cleansed_data = data.withColumn("text_cleaned",regexp_replace(col("text"),"[^a-zA-Z0-9]+", " "))
  except Exception as e:
          logger.exception("Unable to read training & test data. Error: %s", e)


