```python
import mlflow

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression

  

from metrics import score_model

from config import config

from data import get_data

  
  

def train(model, x_train, y_train) -> None:

model.fit(x_train, y_train)

  
  

def test(model, x_test, y_test) -> None:

y_pred = model.predict(x_test)

y_proba = model.predict_proba(x_test)

# Здесь необходимо получить метрики и логировать их в трекер

metrics = score_model(y_test, y_pred, y_proba)

confusion_matrix = metrics.pop("confusion_matrix")

confusion_matrix = pd.DataFrame(

confusion_matrix, index=list(range(10)), columns=list(range(10))

)

fig = plt.figure()

sns.heatmap(confusion_matrix, annot=True)

mlflow.log_metrics(metrics)

mlflow.log_figure(fig, "confusion_matrix.png")

plt.close(fig)

  
  

if __name__ == "__main__":

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

mlflow.set_experiment("Lesson 4")

  

logistic_regression_model = LogisticRegression(

max_iter=config["logistic_regression"]["max_iter"],

)

  

data = get_data()

train(logistic_regression_model, data["x_train"], data["y_train"])

  

with mlflow.start_run(run_name="model1.py"):

signature = infer_signature(

data["x_test"],

logistic_regression_model.predict(data["x_test"]),

)

params = logistic_regression_model.get_params()

mlflow.log_params(

{

"solver": params["solver"],

"C": params["C"],

"coef_": logistic_regression_model.coef_.tolist(),

"intercept_": logistic_regression_model.intercept_.tolist(),

}

)

test(logistic_regression_model, data["x_test"], data["y_test"])

mlflow.sklearn.log_model(

sk_model=logistic_regression_model,

artifact_path="logistic_regression",

signature=signature,

registered_model_name="sklearn_logistic_regression",

)
```