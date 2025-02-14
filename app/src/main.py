import os
import joblib
import json
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


app = FastAPI()
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "../..")
model: BaseEstimator = None


@app.on_event("startup")
def startup():
    os.system(f"cd {PROJECT_ROOT} && dvc pull models/main.dvc")
    global model
    model = joblib.load(PROJECT_ROOT + "/models/main/model.pkl")


@app.post("/predict")
def predict(data: WineFeatures) -> dict:
    # Загрузка модели и получение предсказания
    x = pd.DataFrame([data.model_dump()])
    return {"predicted_quality": model.predict(x)[0]}


@app.get("/health")
def health() -> dict:
    # Проверка работоспособности сервиса
    if model is None:
        return {"status": "error", "reason": "model"}
    else:
        return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    # Информация о модели и её метриках
    with open(PROJECT_ROOT + "/models/main/metadata.json", "r") as file:
        info = json.load(file)

    # Возвращаем информацию о модели
    return info


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
