from pyspark.sql.functions import regexp_replace,col
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
import mlflow
import mlflow.spark
import mlflow.pyfunc
import pyspark.ml.attribute


spark=(SparkSession.builder \
                        .config("spark.hadoop.yarn.timeline-service.enabled","false") \
                        .config("spark.submit.deployMode","client") \
                        .config("spark.executor.instances","10") \
                        .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0") \
                        .config("spark.jars","/home/alexey_kvashnin/gcs-connector-hadoop3-2.2.2.jar") \
                        .master("yarn") \
                        .getOrCreate())

data = spark.read.load("/user/alexey_kvashnin/inputData.parquet")
cleansed_data = data.withColumn("text_cleaned",regexp_replace(col("text"),"[^a-zA-Z0-9]+", " "))

model=mlflow.spark.load_model("models:/Complaint classification model/1")
lrModel=model.stages[-2]

predictions=model.transform(cleansed_data)
schema=predictions.schema

print(schema["features"].metadata)

features_arr=predictions.select(vector_to_array(col("features")).alias("features_arr"))

features_arr.select((0 until 60000).map(i => $"features_arr".getItem(i).as(s"col$i")): _*)


