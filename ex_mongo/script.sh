docker pull mongo:latest

docker run -d -p 27017:27017 --name my_mongo mongo:latest

echo "insertion de données"

python3 feed_bdd.py 

echo "lecture des données"

python3 read_bdd.py

echo "Arrêter le conteneur"

docker stop my_mongo

echo "Rallumer le conteneur"

docker start my_mongo 

echo "Interagir avec le conteneur"

python3 read_bdd.py

echo "Arrêter le conteneur"

docker stop my_mongo

echo "Supprimer le conteneur"

docker rm my_mongo
