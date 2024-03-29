import mlflow
import mlflow.spark
import logging
from pyspark.sql import SparkSession
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF,Tokenizer,StringIndexer,RegexTokenizer,IDF,StopWordsRemover,NGram,VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import regexp_replace,col


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)




if __name__ == "__main__":



 spark = (SparkSession.builder \
             .config("spark.hadoop.yarn.timeline-service.enabled","false") \
             .config("spark.submit.deployMode","client") \
             .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0") \
             .config("spark.executor.instances","10") \
             .master("yarn") \
             .getOrCreate())

 mlflow.set_registry_uri("http://localhost:8000")
 mlflow.set_tracking_uri("http://localhost:8000")


 try:
   mlflow.spark.autolog()
   data = spark.read.load("/user/alexey_kvashnin/inputData.parquet")
   cleansed_data = data.withColumn("text_cleaned",regexp_replace(col("text"),"[^a-zA-Z0-9]+", " "))
 except Exception as e:
         logger.exception(
             "Unable to read training & test data. Error: %s", e
         )

 train, test = cleansed_data.randomSplit([0.75, 0.25], seed=12345)

 with mlflow.start_run() as active_run:

    indexer = StringIndexer(inputCol="Product", outputCol="label")
    tokenizer = RegexTokenizer(inputCol="text_cleaned",outputCol="words",toLowercase=True,minTokenLength=3)
    swr=StopWordsRemover(inputCol="words",outputCol="words_swr")
    tf=HashingTF(numFeatures=30000,inputCol="words_swr",outputCol="tf_out")
    tf2=HashingTF(numFeatures=30000,inputCol="ngram_2",outputCol="tf_out_2")
    idf=IDF(minDocFreq=3,inputCol="tf_out",outputCol="idf_out")
    idf2=IDF(minDocFreq=3,inputCol="tf_out_2",outputCol="idf_out_2")
    lr = LogisticRegression(maxIter=10)
    ngram=NGram(n=2,inputCol="words_swr",outputCol="ngram_2")
    assembler=VectorAssembler(inputCols=["idf_out","idf_out_2"],outputCol="features")



    pipeline=Pipeline(stages=[indexer,tokenizer,swr,ngram,tf,idf,tf2,idf2,assembler,lr])
    model = pipeline.fit(train)

    predictions=model.transform(test)
    evaluator=MulticlassClassificationEvaluator(metricName="weightedFMeasure")

    f1=evaluator.evaluate(predictions)

    mlflow.spark.save_model(model,"spark-model",mflow_model="spark-model")
    mlflow.spark.log_model(model,"spark-model")
    mlflow.log_metric("f1_weighted",f1)






