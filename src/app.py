from fastapi import FastAPI, Request
import joblib
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import logging
import pandas as pd
from fastapi.templating import Jinja2Templates

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

templates = Jinja2Templates("templates/")

class UserData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int  # Обязательное поле
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float  # Обязательное поле
    TotalCharges: float  # Обязательное поле


# Загрузка модели
def load_model():
    try:
        model = joblib.load('models/model.joblib')
        return model
    except Exception as e:
        logger.error('FAILed to load model!')
        return None, None


model = load_model()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "Главная страница"}
    )


@app.post("/predict")
async def predict(data: UserData):
    if model is None:
        return JSONResponse({'error': 'Модель не загружена'}, 500)

    data_dict = data.model_dump()
    
    logger.debug(data_dict)

    input_data = pd.DataFrame([{
        'gender': data_dict['gender'],
        'SeniorCitizen': int(data_dict['SeniorCitizen']),
        'Partner': data_dict['Partner'],
        'Dependents': data_dict['Dependents'],
        'tenure': int(data_dict['tenure']),
        'PhoneService': data_dict['PhoneService'],
        'MultipleLines': data_dict['MultipleLines'],
        'InternetService': data_dict['InternetService'],
        'OnlineSecurity': data_dict['OnlineSecurity'],
        'OnlineBackup': data_dict['OnlineBackup'],
        'DeviceProtection': data_dict['DeviceProtection'],
        'TechSupport': data_dict['TechSupport'],
        'StreamingTV': data_dict['StreamingTV'],
        'StreamingMovies': data_dict['StreamingMovies'],
        'Contract': data_dict['Contract'],
        'PaperlessBilling': data_dict['PaperlessBilling'],
        'PaymentMethod': data_dict['PaymentMethod'],
        'MonthlyCharges': float(data_dict['MonthlyCharges']),
        'TotalCharges': float(data_dict['TotalCharges'])
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return JSONResponse({
        'prediction': 'Yes' if prediction else 'No',
        'probability': float(probability)
    }, 200)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level='debug')
