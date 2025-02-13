
```python
import os

from datetime import timedelta, datetime

  

from airflow import DAG

from airflow.operators.python import PythonOperator

  

from dotenv import load_dotenv

from weather import fetch_weather, save_row

  

default_args = {

"owner": "airflow",

"depends_on_past": False,

"start_date": datetime(2021, 1, 1),

"email_on_failure": False,

"email_on_retry": False,

"retries": 1,

"retry_delay": timedelta(minutes=5),

}

  

load_dotenv(".env")

weather_csv_path = "weather.csv"

api_key = os.getenv("API_KEY")

city = "Moscow"

  

dag = DAG(

"fetch_weather",

default_args=default_args,

description="Выгрузка данных в weather.csv каждую минуту из openweathermap.org",

schedule_interval=timedelta(minutes=1),

catchup=False,

)

  

fetch_and_save_data_task = PythonOperator(

task_id="fetch_and_save_data",

python_callable=save_row,

op_kwargs={"data": fetch_weather(api_key, city), "csv_path": weather_csv_path},

dag=dag,

)
```