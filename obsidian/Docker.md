**3 сервиса**:
- Airflow
- MLflow
- uvicorn

airflow -> mlflow, s3
mlflow -> 
uvicorn -> s3
## Airflow
### DockerFile
```DockerFile
FROM apache/airflow:2.10.4-python3.12

RUN airflow db init

RUN airflow users create --username misha --password 1234 --firstname M --lastname V --role Admin --email mikhail.valiev7@gmail.com

# ENV PYTHONPATH=/opt/airflow/dags

WORKDIR /opt/airflow/dags

COPY --chown=airflow:root . .

RUN pip install -r requirements.txt

ENTRYPOINT ["airflow", "standalone"]
```

```DockerFile
FROM apache/airflow:2.8.3
RUN pip install --no-cache-dir pandas sqlalchemy
```
### docker-compouse.yml
```yml
version: '3'
x-airflow-common:
  &airflow-common
  build: .
  image: apache/airflow:2.8.3 #строчки этой нет во втором примере
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__SCHEDULER__MIN_FILE_PROCESS_INTERVAL: 10
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - .:/app
  user: "${AIRFLOW_UID:-50000}:${AIRFLOW_GID:-50000}"
  depends_on:
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    ports:
      - 5432:5432
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5
    restart: always

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    restart: always

  airflow-init:
    <<: *airflow-common
    command: version
    environment:
      <<: *airflow-common-env
      _AIRFLOW_DB_UPGRADE: 'true'
      _AIRFLOW_WWW_USER_CREATE: 'true'
      _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow}
      _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow}

volumes:
  postgres-db-volume:
```
