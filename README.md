# Оценка качества вина методами машинного обучения

## Структура проекта

```plaintext
project_root/
│
├── data/
│
├── models/
|
├── airflow/
|   ├── config/                         # config and config.local for dvc
│   ├── dags/
|   |   ├── gb_train.py                 # train_test model
│   |   └── dag.py
|   ├── DockerFile
│   └── requirements.txt
|
├── mlflow/
│   ├── src/
│   │   ├── tracking_best_model.py
│   |   └── tracking_experiments.py
|   ├── DockerFile
│   └── requirements.txt
|
├── app/
|   ├── config/                         # config and config.local for dvc
│   ├── src/
│   |   └── main.py
|   ├── DockerFile
│   └── requirements.txt
|
├── tests/
|
├── requirements.txt
|
├── docker-compose.yml
|
├── .env
│
├── .gitlab-ci.yml
│
├── .gitignore
│
├── obsidian/
|
└── README.md
```

## Процесс и результаты экспериментов

В ходе работы были протестированы две модели из библиотеки scikit-learn: GradientBoostingRegressor и RandomForestRegressor. Эксперименты логировались с помощью MLflow. Перебирались следующие гиперпараметры:

```python
RandomForestRegressor:
    {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "min_samples_split": [5, 10],
    "bootstrap": [True, False],
    "criterion": ["squared_error", "absolute_error"],
    }
```

```python
GradientBoostingRegressor:
{
    "n_estimators": [50, 100],
    "learning_rate": [0.1, 0.2],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
    "loss": ["squared_error", "absolute_error"],
}
```

Таблица с наилучшими гиперпараметрами обеих моделей и среднеквадратическая ошибка

| **Run Name**      | random_forest_regressor                                    | gradient_boosting_regressor                                |
| :---------------- | :--------------------------------------------------------- | :--------------------------------------------------------- |
| **Parameters**    |                                                            |                                                            |
| bootstrap         | True                                                       |                                                            |
| criterion         | squared_error                                              |                                                            |
| learning_rate     |                                                            | 0.1                                                        |
| loss              |                                                            | squared_error                                              |
| max_depth         | 10                                                         | 5                                                          |
| min_samples_split | 5                                                          |                                                            |
| n_estimators      | 100                                                        | 100                                                        |
| subsample         |                                                            | 0.8                                                        |
| random_state      | 42                                                         | 42                                                         |
| refit             | neg_root_mean_squared_error                                | neg_root_mean_squared_error                                |
| scoring           | ["neg_root_mean_squared_error", "neg_mean_absolute_error"] | ["neg_root_mean_squared_error", "neg_mean_absolute_error"] |
| **Metrics**       |                                                            |                                                            |
| **RMSE**          | **0.565**                                                  | **0.575**                                                  |

Как мы видим, разница в качестве у моделей невелика, но все же есть. Поэтому для предсказаний в продакшене мы будем использовать **GradientBoostingRegressor**.

## Инструкция по запуску

1. Подключиться к minio s3 и загрузить данные и модель;
2. Запустить docker-compose
   - API прослушивается по порту 8080;
   - MLflow по порту 8081;
   - Airflow по порту 8082.

- Все логи от `mlfow` сохраняются в директорию `mlflow`;

- Если данных и модели еще нет в dvc, то надо
  - скачать [данные](https://github.com/aniruddhachoudhury/Red-Wine-Quality/blob/master/winequality-red.csv) в `data/winequality-red.csv`;
  - запустить `mlflow/src/tracking_best_model.py`;
  - сделать `dvc add data/winequality-red.csv models/main`;
  - не забыть поменять `config`-ы для dvc для `airflow` и `api`.

P.S. дико извиняюсь за названия токенов (не знал, что они где-то будут светиться)
