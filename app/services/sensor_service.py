from app.db.database import get_db
from datetime import datetime
from app.edge.processor import process_sensor_data   

async def add_sensor_data(email: str, data: dict):
    db = get_db()

    # 🔹 Ensure required fields (your existing logic)
    data.setdefault("heart_rate", 75)
    data.setdefault("hrv", 50)
    data.setdefault("spo2", 98)
    data.setdefault("temperature", 36.5)
    data.setdefault("steps", 0)

    # 🔥 EDGE PROCESSING (NEW)
    processed_data = process_sensor_data(data)

    # 🔹 Add metadata
    processed_data["email"] = email
    processed_data["timestamp"] = datetime.utcnow()

    # 🔹 Store in DB
    await db.sensor_data.insert_one(processed_data)

    return {
        "message": "Sensor data processed & stored",
        "alerts": processed_data.get("alerts", [])   # 🔥 return alerts
    }