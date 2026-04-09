import pickle
import numpy as np
from app.db.database import get_db
import os

# 🔥 Get base directory (Backend/app/services/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 Move to Backend/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# 🔥 Model folder path
MODEL_DIR = os.path.join(ROOT_DIR, "Model")

# ✅ Load models
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "mlp_model.pkl"), "rb") as f:
    mlp_model = pickle.load(f)


# 🔹 Aggregate sensor data
def aggregate_sensor_data(records):
    if not records:
        return {
            "heart_rate": 75,
            "hrv": 50,
            "spo2": 98,
            "temperature": 36.5,
            "step_count": 5000,
            "temp_change": 0
        }

    heart_rates = [r["heart_rate"] for r in records]
    hrvs = [r["hrv"] for r in records]
    spo2s = [r["spo2"] for r in records]
    temps = [r["temperature"] for r in records]
    steps = [r["steps"] for r in records]

    return {
        "heart_rate": sum(heart_rates) / len(heart_rates),
        "hrv": sum(hrvs) / len(hrvs),
        "spo2": sum(spo2s) / len(spo2s),
        "temperature": sum(temps) / len(temps),
        "step_count": sum(steps),
        "temp_change": temps[-1] - temps[0] if len(temps) > 1 else 0
    }


async def predict_heart_disease(email: str):
    db = get_db()

    # ✅ Get static data
    profile = await db.profiles.find_one({"email": email})
    if not profile:
        return None
    profile.pop("_id")

    # ✅ Get dynamic data (last 10 records)
    sensor_records = await db.sensor_data.find(
        {"email": email}
    ).sort("timestamp", -1).limit(10).to_list(length=10)

    agg = aggregate_sensor_data(sensor_records)

    # ✅ Derived feature
    pulse_pressure = profile["systolic_bp"] - profile["diastolic_bp"]

    # ✅ Activity level from steps
    if agg["step_count"] < 3000:
        activity_level = 0
    elif agg["step_count"] < 7000:
        activity_level = 1
    else:
        activity_level = 2

    # ✅ FINAL FEATURE VECTOR (MATCH TRAINING ORDER)
    features = [
        profile["age"],
        profile["sex"],
        profile["smoking"],
        profile["diabetes"],
        profile["systolic_bp"],
        profile["diastolic_bp"],
        pulse_pressure,
        profile["cholesterol"],
        profile["bmi"],
        profile["alcohol"],
        agg["heart_rate"],
        agg["hrv"],
        agg["spo2"],
        agg["temperature"],
        agg["temp_change"],
        activity_level,
        agg["step_count"]
    ]

    input_array = np.array([features])

    # ✅ Scale
    input_scaled = scaler.transform(input_array)

    # =========================
    # 🔹 XGBoost
    # =========================
    xgb_pred = xgb_model.predict(input_scaled)[0]
    xgb_prob = (
        xgb_model.predict_proba(input_scaled)[0][1]
        if hasattr(xgb_model, "predict_proba")
        else None
    )

    # =========================
    # 🔹 MLP
    # =========================
    mlp_pred = mlp_model.predict(input_scaled)[0]
    mlp_prob = (
        mlp_model.predict_proba(input_scaled)[0][1]
        if hasattr(mlp_model, "predict_proba")
        else None
    )

    # =========================
    # 🔥 ENSEMBLE
    # =========================
    probs = [p for p in [xgb_prob, mlp_prob] if p is not None]

    if probs:
        final_prob = sum(probs) / len(probs)
    else:
        final_prob = (xgb_pred + mlp_pred) / 2

    final_pred = 1 if final_prob >= 0.5 else 0

    return {
        "prediction": int(final_pred),
        "risk": "High" if final_pred == 1 else "Low",
        "probability": float(final_prob),

        "model_outputs": {
            "xgb": {
                "prediction": int(xgb_pred),
                "probability": float(xgb_prob) if xgb_prob else None
            },
            "mlp": {
                "prediction": int(mlp_pred),
                "probability": float(mlp_prob) if mlp_prob else None
            }
        }
    }