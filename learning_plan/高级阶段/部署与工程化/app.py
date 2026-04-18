
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 加载模型（示例，实际使用时需要训练模型）
# model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # 模拟预测
    features = np.array(request.features).reshape(1, -1)
    # prediction = model.predict(features)[0]
    # confidence = model.predict_proba(features).max()
    
    # 模拟结果
    prediction = 1
    confidence = 0.95
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )

@app.get('/')
def read_root():
    return {"message": "AI Model API"}
