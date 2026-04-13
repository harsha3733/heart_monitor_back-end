import pickle
import numpy as np
from app.db.database import get_db
import os

# ── Path setup ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "Model")

# ── Load models ──────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
    hgb_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "mlp_model.pkl"), "rb") as f:
    mlp_model = pickle.load(f)

# ── Feature order — MUST match training exactly ──────────────────
FEATURE_ORDER = [
    "age", "sex", "smoking", "diabetes",
    "systolic_bp", "diastolic_bp", "pulse_pressure",
    "cholesterol", "bmi", "alcohol",
    "heart_rate", "hrv", "spo2", "temperature",
    "step_count", "activity_level",
]

# ── Training data hard limits (model breaks outside these) ───────
# If sensor sends HRV=500 the MLP collapses to 0.0 and kills the prediction
CLAMP = {
    "age":           (18,   90),
    "systolic_bp":   (80,  220),
    "diastolic_bp":  (50,  130),
    "pulse_pressure":(10,  100),
    "cholesterol":   (100, 400),
    "bmi":           (15,   55),
    "heart_rate":    (45,  130),
    "hrv":           (10,  140),  # raw RR intervals (500+) are NOT HRV — clamp hard
    "spo2":          (88,  100),
    "temperature":   (35,   39),
    "step_count":    (0,  15000),
}


def clamp(value, feature):
    lo, hi = CLAMP[feature]
    return max(lo, min(float(value), hi))


# ── HRV unit conversion ──────────────────────────────────────────
# Many sensors send raw RR interval (ms between beats, ~600-1000ms)
# instead of RMSSD HRV (10-140ms). Detect and convert.
def normalize_hrv(raw_hrv: float) -> float:
    if raw_hrv > 140:
        # Likely a raw RR interval — estimate RMSSD as ~10% of RR
        estimated_hrv = raw_hrv * 0.10
        return clamp(estimated_hrv, "hrv")
    return clamp(raw_hrv, "hrv")


# ── Aggregate the last 30 sensor records ─────────────────────────
def aggregate_sensor_data(records):
    if not records:
        return {
            "heart_rate": 75.0,
            "hrv": 60.0,
            "spo2": 98.0,
            "temperature": 36.6,
            "step_count": 5000,
        }

    return {
        "heart_rate": sum(r["heart_rate"] for r in records) / len(records),
        "hrv":        sum(r["hrv"]        for r in records) / len(records),
        "spo2":       sum(r["spo2"]       for r in records) / len(records),
        "temperature":sum(r["temperature"]for r in records) / len(records),
        "step_count": sum(r["steps"]      for r in records),
    }


# ── Clinical rule-based score ────────────────────────────────────
# Runs independently alongside the ML model.
# Ensures high-risk clinical profiles are never ignored
# even if sensor data is ambiguous or out of range.
def clinical_risk_score(profile, agg):
    score = 0.0

    if profile["age"] > 55:             score += 0.15
    if profile["age"] > 65:             score += 0.10  # extra for elderly
    if profile["systolic_bp"] > 140:    score += 0.20
    if profile["systolic_bp"] > 160:    score += 0.10  # extra for stage 2
    if profile["diabetes"] == 1:        score += 0.15
    if profile.get("smoking", 0) >= 1:  score += 0.10
    if profile["cholesterol"] > 240:    score += 0.10
    if profile["bmi"] > 30:             score += 0.05
    if agg["heart_rate"] > 90:          score += 0.08
    if agg["hrv"] < 40:                 score += 0.10
    if agg["spo2"] < 95:                score += 0.10

    return min(score, 1.0)


# ── Main prediction function ─────────────────────────────────────
async def predict_heart_disease(email: str):
    db = get_db()

    # 1. Static profile data
    profile = await db.profiles.find_one({"email": email})
    if not profile:
        return None
    profile.pop("_id")

    # 2. Last 30 sensor readings
    sensor_records = await db.sensor_data.find(
        {"email": email}
    ).sort("timestamp", -1).limit(30).to_list(length=30)

    agg = aggregate_sensor_data(sensor_records)

    # 3. Normalize HRV — convert raw RR interval to RMSSD if needed
    agg["hrv"] = normalize_hrv(agg["hrv"])

    # 4. Clamp all values to training ranges
    heart_rate   = clamp(agg["heart_rate"],        "heart_rate")
    hrv          = clamp(agg["hrv"],               "hrv")
    spo2         = clamp(agg["spo2"],              "spo2")
    temperature  = clamp(agg["temperature"],       "temperature")
    step_count   = clamp(agg["step_count"],        "step_count")
    age          = clamp(profile["age"],            "age")
    systolic_bp  = clamp(profile["systolic_bp"],    "systolic_bp")
    diastolic_bp = clamp(profile["diastolic_bp"],   "diastolic_bp")
    cholesterol  = clamp(profile["cholesterol"],    "cholesterol")
    bmi          = clamp(profile["bmi"],            "bmi")
    pulse_pressure = clamp(systolic_bp - diastolic_bp, "pulse_pressure")

    # 5. Normalize smoking/alcohol: profile allows 0/1/2, model trained on 0/1
    smoking = 1 if profile.get("smoking", 0) >= 1 else 0
    alcohol  = 1 if profile.get("alcohol",  0) >= 1 else 0

    # 6. Activity level from clamped step count
    if step_count < 3000:
        activity_level = 0
    elif step_count < 7000:
        activity_level = 1
    else:
        activity_level = 2

    # 7. Build feature vector
    features = [
        age, profile["sex"], smoking, profile["diabetes"],
        systolic_bp, diastolic_bp, pulse_pressure,
        cholesterol, bmi, alcohol,
        heart_rate, hrv, spo2, temperature,
        step_count, activity_level,
    ]

    # 8. Scale and predict
    input_scaled = scaler.transform(np.array([features]))

    hgb_prob = hgb_model.predict_proba(input_scaled)[0][1]
    mlp_prob = mlp_model.predict_proba(input_scaled)[0][1]
    model_prob = 0.6 * hgb_prob + 0.4 * mlp_prob

    # 9. Clinical rule-based score (parallel to model)
    agg_normalized = {**agg, "heart_rate": heart_rate, "hrv": hrv, "spo2": spo2}
    clinical_prob = clinical_risk_score(profile, agg_normalized)

    # 10. Final hybrid probability — 50% model + 50% clinical rules
    final_prob = 0.5 * model_prob + 0.5 * clinical_prob
    final_pred = 1 if final_prob >= 0.5 else 0

    # 11. Risk tier
    if final_prob >= 0.65:
        risk_level = "High"
    elif final_prob >= 0.35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "prediction": int(final_pred),
        "risk": risk_level,
        "probability": round(float(final_prob), 4),

        "model_outputs": {
            "hgb":             {"probability": round(float(hgb_prob), 4)},
            "mlp":             {"probability": round(float(mlp_prob), 4)},
            "model_ensemble":  round(float(model_prob), 4),
            "clinical_score":  round(float(clinical_prob), 4),
        },

        "input_summary": {
            "heart_rate":     round(heart_rate, 1),
            "hrv":            round(hrv, 1),
            "hrv_raw":        round(agg["hrv"], 1),
            "spo2":           round(spo2, 1),
            "temperature":    round(temperature, 1),
            "step_count":     int(step_count),
            "activity_level": activity_level,
            "pulse_pressure": int(pulse_pressure),
        },
    }