Sistema recomendador usando un dataset similar al de Movielens pero usando CSVs de animes y utilizando Apache Spark en la ejecución del algoritmo ALS.

Faltan los CSVs de ratings totales, ya que su tamaño es muy grande y no se pueden subir a GitHub

## venv

Necesitas la herramienta virtualenv `pip install virtualenv`

Crea un entorno virtual `virtualenv venv`

Actívalo en Windows`venv\scripts\activate`

## Dependencies

Instala dependencias `pip install -r requirements.txt`

Si quieres actualizar dependencias:
Necesitas la herramienta pipreqs `pip install pipreqs`

`pipreqs --force .`

## Run

Ejecuta spark `python app.py`
