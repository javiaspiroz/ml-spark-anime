import pandas as pd
from pyspark.ml import recommendation
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

def guardar(df, name):
    text_file = open('output/'+name+'.txt','w', encoding='utf8')
    text_file.write(df.to_string())
    text_file.close()
    html = df.to_html()
    html_file =open('output/'+name+".html", 'w',encoding='utf8')
    html_file.write(html)
    html_file.close()

pd.set_option("display.max_columns", None)
spark = SparkSession.builder.master("local[*]").getOrCreate()

animes = spark.read.csv("dataset_valoraciones_anime/anime.csv", header=True, inferSchema=True, sep=",", encoding="utf8",escape="\"")
# animes.show()
ratings = spark.read.csv("dataset_valoraciones_anime/rating_red.csv", header=True, inferSchema=True, sep=",")
# ratings.show()

# Dividimos los dataframes en test y training
(training,test) = ratings.randomSplit([0.8, 0.2])
# Entrenamos el modelo, se usa coldstart con drop para descartar Nan
als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
model=als.fit(training)
# Evaluamos el modelo con RMSE
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
users = ratings.filter(ratings["user_id"]==666666)
userSubsetRecs = model.recommendForUserSubset(users, 100)
resultado = userSubsetRecs.first()['recommendations']
recommendations=[]
for i in resultado:
    recommendations.append(i['anime_id'])
dataframe=spark.createDataFrame(pd.DataFrame(recommendations, columns=["Anime_ID"]))
result_ratings = dataframe.join(animes, [dataframe.Anime_ID==animes.ID], 'left').select('ID','Name','Japanese name','Type')
results=result_ratings.drop('Anime_ID')

peliculas=results.filter(results['Type']=="Movie").toPandas()
peliculas = peliculas[0:5]
print(peliculas)
guardar(peliculas, "peliculas")

series=results.filter(results['Type']=="TV").toPandas()
series = series[0:5]
print(series)
guardar(series, "series")
