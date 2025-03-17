from fastapi import FastAPI
import uvicorn

from irisModel import IrisMuchineLearning, IrisSpecies

app = FastAPI()

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
    pred = model.predict_species(iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width)
    return {"prediction": pred}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)