```python
import uvicorn

import joblib

import os

import pandas as pd

from sklearn.base import BaseEstimator

from fastapi import FastAPI

from pydantic import BaseModel

  

app = FastAPI()

model: BaseEstimator = None

  
  

class InputFeatures(BaseModel):

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

  
  

@app.on_event("startup")

def startup():

os.system("dvc pull")

global model

if os.path.exists("model.pkl"):

model = joblib.load("model.pkl")

  
  

@app.post("/predict")

async def predict(x: InputFeatures):

x = pd.DataFrame([x.model_dump()])

print(model.predict(x))

return {"predicted_quality": model.predict(x)[0]}

  
  

@app.get("/healthcheck")

async def healthcheck():

if model is None:

return {"status": "error", "reason": "model"}

else:

return {"status": "ok"}

  
  

if __name__ == "__main__":

uvicorn.run(app, host="127.0.0.1", port=8000)
```