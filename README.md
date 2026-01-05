Fuentes (APIs · Logs · DB)
        ↓
     Kafka  ←────────────── SNMP (brokers, lag, interfaces)
        ↓
Spark (Streaming / Batch) ← SNMP (nodos, CPU, memoria)
        ↓
MinIO  ←→  Iceberg        ← SNMP (estado del servicio, I/O)
        ↓           ↑
      Trino      Hive Metastore (prescindible en esta versión)
        ↓              ↑
     BI / Analytics   SNMP (servicios, latencia)

Airflow  → Orquestación
Grafana  → Monitorización
Prometheus → Métricas (prescindible en esta versión)
SNMP Manager → Gestión de red (FCAPS)


Airbyte es como kafka pero para apis y dbs
dbt transformación

Para ML solo quitar Kafka y Streaming 

 
EJECUTAR EN LA CARPETA DOCKER
docker compose pull --parallel=false
docker compose --env-file ../.env up --pull=always 

docker compose down -v
docker volume ls
docker compose logs -f backend
docker compose ps
docker compose exec NOMBRE bash


