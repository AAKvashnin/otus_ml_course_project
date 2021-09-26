import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
import pandas as pd


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def main():


  spark=(SparkSession.builder \
                        .config("spark.hadoop.yarn.timeline-service.enabled","false") \
                        .config("spark.submit.deployMode","client") \
                        .config("spark.executor.instances","4") \
                        .master("yarn") \
                        .getOrCreate())

  model=mlflow.spark.load_model("models:/Complaint classification model/1")


  pandasDF=pd.read_csv("data/data.txt")

  sparkDF=spark.createDataFrame(pandasDF)

  predictions=model.transform(sparkDF);









