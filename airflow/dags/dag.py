from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from gb_train import train_test_log

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 9),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "TrainModel",
    default_args=default_args,
    description="Ежедневное обучение модели",
    schedule_interval=timedelta(days=1),
    catchup=False,
)


load_data_task = BashOperator(
    task_id="dvc_pull", bash_command="cd $PROJECT_ROOT && dvc pull", dag=dag
)

train_model_task = PythonOperator(
    task_id="train_test_log", python_callable=train_test_log, dag=dag
)

push_data_task = BashOperator(
    task_id="dvc_push_model",
    bash_command="""cd $PROJECT_ROOT && 
        dvc add models/main && dvc push models/main.dvc""",
    dag=dag,
)

load_data_task >> train_model_task >> push_data_task
