import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


file_obj = open("anime_spark_ml\dataset_valoraciones_anime\/anime.csv", encoding="utf-8")
master = pd.read_csv(file_obj)
print(master)