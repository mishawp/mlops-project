import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    project_root = os.getenv("PROJECT_ROOT", "../..")

    mlflow.set_tracking_uri("http://127.0.0.1:8081")
    mlflow.set_experiment("Experiments")

    data = pd.read_csv(project_root + "/data/winequality-red.csv")

    X = data.drop("quality", axis=1)
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    custom_cv = [(X_train.index, X_test.index)]

    gradient_boosting_params = {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 0.2],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "loss": ["squared_error", "absolute_error"],
    }

    random_forest_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [5, 10],
        "bootstrap": [True, False],
        "criterion": ["squared_error", "absolute_error"],
    }

    gradient_boosting_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gradient_boosting_params,
        cv=custom_cv,
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error"],
        refit="neg_root_mean_squared_error",
    )

    random_forest_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        random_forest_params,
        cv=custom_cv,
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error"],
        refit="neg_root_mean_squared_error",
    )

    mlflow.sklearn.autolog(
        log_model_signatures=True, log_models=True, max_tuning_runs=10
    )

    with mlflow.start_run(run_name="gradient_boosting_regressor"):
        gradient_boosting_search.fit(X, y)

    with mlflow.start_run(run_name="random_forest_regressor"):
        random_forest_search.fit(X, y)
