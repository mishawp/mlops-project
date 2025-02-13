```python
import os

from datetime import timedelta, datetime

  

import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from airflow import DAG

from airflow.operators.python import PythonOperator

from data import load_data, prepare_data

from train import train

from test import test

  
  

default_args = {

'owner': 'airflow',

'depends_on_past': False,

'start_date': datetime(2021, 1, 1),

'email_on_failure': False,

'email_on_retry': False,

'retries': 1,

'retry_delay': timedelta(minutes=5),

}

  

dataset_path = "/opt/airflow/dags/dataset"

  

dag = DAG(

'LogReg_iris',

default_args=default_args,

description='A simple DAG to classify irises',

schedule_interval=timedelta(days=1),

)

  

load_data_task = PythonOperator(

task_id='load_data',

python_callable=load_data,

dag=dag,

)

  

prepare_data_task = PythonOperator(

task_id='prepare_data',

python_callable= prepare_data,

op_kwargs={"csv_path": dataset_path + "/iris.csv"},

dag=dag,

)

  

train_task = PythonOperator(

task_id='train',

python_callable= train,

op_kwargs={"train_csv": dataset_path + "/iris_train.csv"},

dag=dag,

)

  

test_task = PythonOperator(

task_id='test',

python_callable= test,

op_kwargs={"model_path": "/opt/airflow/dags/model.pkl", "test_csv": dataset_path + "/iris_test.csv"},

dag=dag,

)

  

load_data_task >> prepare_data_task >> train_task >> test_task
```