import mlflow
import pandas as pd
import os
import joblib
import json
from datetime import date
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error


def train_test_log():
    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:8081"
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Main model")

    project_root = os.getenv("PROJECT_ROOT", "../..")
    data_path = project_root + "/data/winequality-red.csv"
    model_dir = project_root + "/models/main"
    model_pkl = model_dir + "/model.pkl"

    data = pd.read_csv(data_path)
    X = data.drop("quality", axis=1)
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    params = dict(
        random_state=42,
        learning_rate=0.1,
        loss="squared_error",
        max_depth=5,
        n_estimators=100,
        subsample=0.8,
    )

    model = GradientBoostingRegressor(**params)

    model.fit(X_train, y_train)
    cur_date = str(date.today())
    with mlflow.start_run(
        run_name=cur_date + "-" + str(model.__class__.__name__)
    ):
        mlflow.log_params(params)
        rmse = root_mean_squared_error(y_test, model.predict(X_test))
        mlflow.log_metric("RMSE", rmse)

        signature = infer_signature(
            X_test.iloc[:10], model.predict(X_test.iloc[:10])
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=cur_date,
            signature=signature,
            registered_model_name="gradient_boosting_regressor",
        )

    metadata = {
        "model_name": model.__class__.__name__,
        "hyperparameters": params,
        "metrics": {"RMSE": rmse},
        "date": cur_date,
    }
    joblib.dump(model, model_pkl)
    with open(model_dir + "/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    train_test_log()
