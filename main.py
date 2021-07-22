import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import status
from fastapi import FastAPI
from pydantic import BaseModel

from app.app import *

app = FastAPI()

topic_data = pd.read_csv("data/topic_data.csv")

@app.get("/")
def hello():
    return {"message":"Hello Sherlock"}

@app.get('/top_10_clusters')
async def top_10_clusters():
    top_10_labels = [0, 3, 2, 5, 34, 33, 18, 22, 12, 41]
    LDA_model = joblib.load('models/LDA_models_50.pkl')
    
    results = [str(top_10_words(LDA_model, label)) for label in top_10_labels]
    res = "-".join(results)
    return res

@app.get('/get_score')
async def get_score(text: str):
    scores = calculate_score(text)
    res = str(list(scores.values())).strip('[]')
    return res

@app.get('/similar_tickets')
async def similar_tickets(label: int, nb_tickets: int):
    return topic_data[topic_data.topic == label][:nb_tickets][['Number', 'Description']]

@app.get('/LDA_predict')
async def LDA_predict(model_name: str, data: str):
    embedded_data = preprocess(data)
    LDA_model = joblib.load('models/LDA_models_50.pkl')
    prediction = LDA_model.transform(embedded_data).argmax(axis=1)[0]
    
    top10Words = top_10_words(LDA_model, prediction)
    top10Words = ' '.join(top10Words)
        
    result = str(prediction) + " - " + top10Words

    return result


@app.get('/predict')
async def predict(model_name: str, data: str):
    embedded_data = preprocess(data)
    embedded_data = np.atleast_2d(embedded_data)
    
    model = joblib.load('models/{}'.format(model_name))
    
    prediction = str(model.predict(embedded_data)[0])
    
    
    return prediction


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

"""
To cover:
- possible return types: not nympy array, etc
"""