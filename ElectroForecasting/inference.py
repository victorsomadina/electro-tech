from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
from datetime import date

app = FastAPI()

class ForecastRequest(BaseModel):
    category: str
    start_date: date
    end_date: date

def preprocessing_pipeline(df):
    df['ds'] = pd.to_datetime(df['ds'])
    df['ds'] = df['ds'].dt.to_period('M').dt.to_timestamp()
    return df

def load_model(category):
    with open(f'Model Training/{category}_model.pkl', 'rb') as file:
        model, _ = pickle.load(file)
    return model

@app.post("/predict")
def predict(request: ForecastRequest):
    df = pd.DataFrame({'ds': pd.date_range(request.start_date, request.end_date)})
    df = preprocessing_pipeline(df)
    
    model = load_model(request.category)
    prediction = model.predict(df)
    
    return prediction[['ds', 'yhat']].to_dict(orient='records')

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=80)
