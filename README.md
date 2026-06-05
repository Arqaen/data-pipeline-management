# Data Pipeline Management

Proyecto para levantar una plataforma local de datos con Docker. La idea
principal es simular un pipeline moderno donde se generan eventos, se envian a
Kafka, se procesan con Spark desde Airflow y se guardan en MinIO.

Tambien hay una parte separada de modelos financieros, usada para analizar datos
historicos, entrenar modelos y generar metricas.

## Que hace el proyecto

- Genera eventos de ejemplo con Python.
- Publica esos eventos en Kafka.
- Airflow organiza la ejecucion del pipeline.
- Spark lee los eventos de Kafka y los guarda en MinIO.
- La capa `bronze` ya esta implementada.
- Las capas `silver` y `gold` estan preparadas, pero todavia no tienen logica.

## Servicios principales

| Servicio | Para que sirve |
| --- | --- |
| Airflow | Organiza y lanza el pipeline. |
| Spark | Procesa los datos. |
| Kafka | Recibe los eventos generados. |
| MinIO | Guarda los datos procesados. |
| Postgres | Guarda la informacion interna de Airflow. |
| Zookeeper | Servicio de apoyo para Kafka. |

## Estructura del repo

```text
airflow/    DAGs de Airflow
docker/     Docker Compose y Dockerfiles
kafka/      productor de eventos
spark/      jobs de Spark
models/     analisis y modelos financieros
sql/        scripts auxiliares para MinIO
data/       datos locales generados por MinIO
test/       resultados y pruebas locales
```

## Archivos importantes

### `docker/docker-compose.yml`

Levanta toda la plataforma:

- Airflow
- Postgres
- Kafka
- Zookeeper
- Spark master y workers
- MinIO
- Productores de eventos

### `airflow/dags/lakehouse_pipeline.py`

Define el pipeline de Airflow. Actualmente lanza el job de Spark que procesa la
capa `bronze`.

### `kafka/kafka_producer.py`

Genera eventos aleatorios de ejemplo y los envia al topic `events` de Kafka.

Ejemplo de evento:

```json
{
  "user_id": 12,
  "product": "A",
  "price": 45.6,
  "timestamp": 1710000000.0
}
```

### `spark/spark_bronze.py`

Lee eventos desde Kafka, los filtra por ventana de tiempo y los guarda en MinIO
en formato Parquet.

Salida principal:

```text
s3a://bronze/eventos_batch
```

### `spark/spark_silver.py` y `spark/spark_gold.py`

Archivo preparado para la futura capa `silver`. Actualmente esta vacio.

### `models/get_data.py`

Script para descargar datos financieros.

### `models/predictions.py`

Script independiente del pipeline principal. Usa datos financieros para entrenar
modelos, evaluar resultados y generar graficos.

# COMANDOS

## Ejecutar

```bash
docker compose up --pull=always --build
```

## Ejecutar en la carpeta docker

```bash
docker compose pull --parallel=false
docker compose --env-file ../.env up --pull=always --build
```

## Parar

```bash
docker compose stop
```

## Alternativa

```bash
docker compose down
```

## No usar

```bash
docker compose down -v --remove-orphans
```

## Consultar

```bash
docker volume ls
docker compose logs -f backend
docker compose ps
```

## Entrar en un contenedor

```bash
docker compose exec NOMBRE bash
```

## Ver password de Airflow

```bash
docker compose exec -it airflow cat simple_auth_manager_passwords.json.generated
```
