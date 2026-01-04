Fuentes (APIs · Logs · DB)
        ↓
     Kafka  ←────────────── SNMP (brokers, lag, interfaces)
        ↓
Spark (Streaming / Batch) ← SNMP (nodos, CPU, memoria)
        ↓
MinIO  ←→  Iceberg        ← SNMP (estado del servicio, I/O)
        ↓           ↑
      Trino      Hive Metastore
        ↓              ↑
     BI / Analytics   SNMP (servicios, latencia)

Airflow  → Orquestación
Grafana  → Monitorización
Prometheus → Métricas
SNMP Manager → Gestión de red (FCAPS)


Airbyte es como kafka pero para apis y dbs
dbt transformación

Para ML solo quitar Kafka y Streaming