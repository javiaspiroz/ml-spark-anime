import pandas as pd
from pyspark.ml import recommendation
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.master("local[*]").getOrCreate()

animes = spark.read.csv("dataset_valoraciones_anime/anime.csv", header=True, inferSchema=True, sep=",", encoding="utf8",escape="\"")
# animes.show()
ratings = spark.read.csv("dataset_valoraciones_anime/rating_red.csv", header=True, inferSchema=True, sep=",")
# ratings.show()

# 1ª version separando pelis y TV al principio y entrenando por separado

# Unimos los dos CSVs para saber el tipo de anime
ratings = ratings.join(animes,ratings.anime_id==animes.ID,"inner").select(ratings["*"],animes["Type"])
ratings.show()
# Separamos los ratings en pelis y series
ratingsTV= ratings.filter(ratings['Type']=="TV")
ratingsTV.show()
ratingsMovies= ratings.filter(ratings['Type']=="Movie")
ratingsMovies.show()
ratings = [ratingsMovies,ratingsTV]

i=0 # Contador para escribir el fichero final
for df in ratings:
    # Dividimos los dataframes en test y training
    (training,test) = df.randomSplit([0.8, 0.2])
    # Entrenamos el modelo, se usa coldstart con drop para descartar Nan
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
    model=als.fit(training)
    # Evaluamos el modelo con RMSE
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    users = df.filter(df["user_id"]==666666)
    userSubsetRecs = model.recommendForUserSubset(users, 5)
    movies=userSubsetRecs.first()['recommendations']
    recommendations=[]
    for movie in movies:
        recommendations.append(movie['anime_id'])
    print(recommendations)
    result = animes.filter((animes.ID).isin(recommendations)).select('ID','Name','Japanese name')
    result.show()

# Añadido porque no sale solo al terminar la ejecución
exit(0)