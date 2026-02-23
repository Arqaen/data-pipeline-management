Fuentes (APIs · Logs · DB)
        ↓
     Kafka 
        ↓
Spark (Streaming / Batch) 
        ↓
MinIO  ←→  Iceberg     
        ↓         
      Trino     
        ↓            
     BI / Analytics   


Airflow  → Orquestación
Grafana  → Monitorización
Prometheus → Métricas (prescindible en esta versión)




Airbyte es como kafka pero para apis y dbs
dbt transformación

Para ML solo quitar Kafka y Streaming 






docker compose up --pull=always --build



 
EJECUTAR EN LA CARPETA DOCKER
docker compose pull --parallel=false
docker compose --env-file ../.env up --pull=always --build



Usa:

docker compose down
o incluso mejor:
docker compose stop
NO:
docker compose down -v --remove-orphans

docker volume ls
docker compose logs -f backend
docker compose ps
docker compose exec NOMBRE bash

docker compose exec -it airflow cat simple_auth_manager_passwords.json.generated
