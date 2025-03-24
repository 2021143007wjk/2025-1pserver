from fastapi import FastAPI
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from irisModel import IrisMuchineLearning, IrisSpecies

app = FastAPI()


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = IrisMuchineLearning()

@app.get("/")
async def root():
    return {"message":"Hello this is iris classfier 25.3.10"}

@app.get("/predict")
async def predict():
    pred = model.predict_species(5.0,3.4,1.4,0.2)
    # pred = "아메리카뒷부리장다리물떼새"
    return {"prediction": pred}

@app.post("/predict")
async def pridict_species(iris:IrisSpecies):
    pred, prob = model.predict_species(iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width)
    return {"prediction": pred, "probability": prob.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)