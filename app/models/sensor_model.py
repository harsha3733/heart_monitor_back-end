from pydantic import BaseModel
from datetime import datetime

class SensorData(BaseModel):
    heart_rate: float
    hrv: float
    spo2: float
    temperature: float
    steps: int
    timestamp: datetime