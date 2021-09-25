import mlflow
import mlflow.spark


from pyspark.sql import SparkSession

spark = (SparkSession.builder \
                .config("spark.hadoop.yarn.timeline-service.enabled","false") \
                .config("spark.submit.deployMode","client") \
                .config("spark.jars","/home/alexey_kvashnin/gcs-connector-hadoop3-2.2.2.jar") \
                .config("spark.executor.instances","10") \
                .master("yarn") \
                .getOrCreate())

model=mlflow.spark.load_model("models:/Complaint classification model/1");
