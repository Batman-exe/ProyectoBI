from typing import Optional, List
import pandas as pd
from fastapi import FastAPI
from joblib import load
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize, StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from tokenizer import tokenizer

from pydantic import BaseModel
class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    text: str

#Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["text"]


# tt = TweetTokenizer()

# def tokenizer(text):
#    return tt.tokenize(text)

from tokenizer import tokenizer
class Model:

    def __init__(self,columns):
        self.model = model

    def make_predictions(self, data):
        result =list()
        for x in data:
            result.append(self.model.predict(x))

        return result

app = FastAPI()



if __name__ == "main":
   model = load("pipeline.joblib")



def make_predictions(dataModel: DataModel):

   feature_list = dict(dataModel)

   df = pd.DataFrame(feature_list, columns=feature_list.keys(), index=[0])
   df.columns = dataModel.columns()
   result = model.predict(df)
   return result.tolist()

@app.post("/predictions")
def make_multiple_predictions(dataModels: List[DataModel]):
   predictions = []
   for i in dataModels:
      result = make_predictions(i)
      predictions.append(result[0])
   return predictions


