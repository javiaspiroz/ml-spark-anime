import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
import time, requests
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("display.max_columns", None) # Para imprimir completamente el datframe de pandas

# Función para crear los archivos .txt y HTML
def guardar(df, name):
    df = df[0:5] # Cogemos solo las 5 1as
    # Creamos listas para almacenar los links obtenidos por la API, unas son para el txt y otras para el HTML con los tags
    images = []
    videos = []
    imagesHTML = []
    videosHTML = []
    htmlDF=pd.DataFrame(df) # Copia del df para añadir a este los tags del HTML

    for i in range(5): #Obtenemos los datos de la API para los animes recomendados
        r = requests.get('https://api.jikan.moe/v3/anime/'+str(df.iloc[i].loc['ID']))
        image=str(r.json()['image_url'])
        # print(image)
        video=str(r.json()['trailer_url'])
        video=video.replace("?enablejsapi=1&wmode=opaque&autoplay=1",'') # Elimino parte sobrante del link como autoplay
        # print(video)
        # Añadimos los links de la API a las listas
        images.append(image)
        videos.append(video)
        # Comprobamos si tiene trailer en la API o no
        imagesHTML.append('<img src="'+image+'" />')
        if video=="None":
            videosHTML.append('<p>Trailer no encontrado</p>')
        else: 
            videosHTML.append('<iframe height="318" width="565.33" src="'+video+'"></iframe>')
        time.sleep(2) # Para evitar el bloqueo de la API
    # Añadimos las nuevas columnas al DF del txt
    df['Image'] = images
    df['Trailer'] = videos
    print(df)
    # Creamos el fichero txt
    txt = open('output/'+name+'.txt', 'w', encoding='utf8')
    txt.write(df.to_string())
    txt.close()
    # Creamos el fichero HTML y añadimos sus columnas al DF
    htmlDF['Image'] = imagesHTML
    htmlDF['Trailer'] = videosHTML
    print(htmlDF)
    htmlDF.to_html("output/{}.html".format(name),escape=False)


spark = SparkSession.builder.master("local[*]").getOrCreate()

# Leo los CSVs
animes = spark.read.csv("dataset_valoraciones_anime/anime.csv", header=True, inferSchema=True, sep=",", encoding="utf8", escape="\"")
# animes.show()
ratings = spark.read.csv("dataset_valoraciones_anime/rating_red.csv", header=True, inferSchema=True, sep=",")
# ratings.show()

# Dividimos los dataframes en test y training
(training,test) = ratings.randomSplit([0.8, 0.2])
# Entrenamos el modelo, se usa coldstart con drop para descartar Nan
# als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)
# Evaluamos el modelo con RMSE
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-squareerror = "+str(rmse))
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
guardar(peliculas, "peliculas")
# Separamos las series
series = recsFull.filter(recsFull['Type']=="TV").toPandas()
guardar(series, "series")
