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
