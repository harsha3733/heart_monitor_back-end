from fastapi import APIRouter, Depends
from app.models.sensor_model import SensorData
from app.services.sensor_service import add_sensor_data
from app.utils.security import get_current_user
from app.db.database import get_db

router = APIRouter(prefix="/sensor", tags=["Sensor"])


# 🔹 1. Add sensor data (existing)
@router.post("/")
async def add_data(
    sensor: SensorData,
    current_user: str = Depends(get_current_user)
):
    return await add_sensor_data(current_user, sensor.dict())


# 🔥 2. Get latest sensor data (FOR DASHBOARD POLLING)
@router.get("/latest")
async def get_latest_sensor(current_user: str = Depends(get_current_user)):
    db = get_db()

    latest = await db.sensor_data.find_one(
        {"email": current_user},
        sort=[("timestamp", -1)]
    )

    if not latest:
        return {}

    latest.pop("_id")

    return latest


# 🔥 3. (Optional) Get recent history (last N records)
@router.get("/history")
async def get_sensor_history(
    limit: int = 10,
    current_user: str = Depends(get_current_user)
):
    db = get_db()

    records = await db.sensor_data.find(
        {"email": current_user}
    ).sort("timestamp", -1).limit(limit).to_list(length=limit)

    for r in records:
        r.pop("_id")

    return records