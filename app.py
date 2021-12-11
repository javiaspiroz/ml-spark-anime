import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

# Función para crear los archivos .txt y HTML
def guardar(df, name):
    txt = open('output/'+name+'.txt', 'w', encoding='utf8')
    txt.write(df.to_string())
    txt.close()
    htmlDF = df.to_html()
    html = open('output/'+name+".html", 'w', encoding='utf8')
    html.write(htmlDF)
    html.close()

pd.set_option("display.max_columns", None) # Para imprimir completamente el datframe de pandas
spark = SparkSession.builder.master("local[*]").getOrCreate()

# Leo los CSVs
animes = spark.read.csv("dataset_valoraciones_anime/anime.csv", header=True, inferSchema=True, sep=",", encoding="utf8", escape="\"")
# animes.show()
ratings = spark.read.csv("dataset_valoraciones_anime/rating_red.csv", header=True, inferSchema=True, sep=",")
# ratings.show()

# Dividimos los dataframes en test y training
(training,test) = ratings.randomSplit([0.8, 0.2])
# Entrenamos el modelo, se usa coldstart con drop para descartar Nan
als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)
# Evaluamos el modelo con RMSE
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
users = ratings.filter(ratings["user_id"]==666666) # Es el ID del usuario EP (666666) para el que recomendamos
userSubsetRecs = model.recommendForUserSubset(users, 100) #Tomamos 100 recomendaciones como nº suficiente par auqe haya 5 pelis y 5 series
userRecs = userSubsetRecs.first()['recommendations']
# Almacenamos las recomendaciones
recs = []
for i in userRecs:
    recs.append(i['anime_id'])

# Creamos un DF con el que unimos los datos de ambos CSVs
df=spark.createDataFrame(pd.DataFrame(recs, columns=["Anime_ID"]))
userRecsRats = df.join(animes, [df.Anime_ID==animes.ID], 'left').select('ID', 'Name', 'Japanese name', 'Type')
recsFull = userRecsRats.drop('Anime_ID')
# Separamos las películas
peliculas=recsFull.filter(recsFull['Type']=="Movie").toPandas()
peliculas = peliculas[0:5] # Cogemos solo las 1as
print(peliculas)
guardar(peliculas, "peliculas")
# Separamos las series
series = recsFull.filter(recsFull['Type']=="TV").toPandas()
series = series[0:5] # Cogemos solo las 1as
print(series)
guardar(series, "series")
