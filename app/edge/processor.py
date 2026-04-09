from app.edge.filters import filter_noise
from app.edge.anomaly import detect_anomalies


def process_sensor_data(data: dict):
    """
    Full edge pipeline:
    - Filtering
    - Feature engineering
    - Anomaly detection
    """

    # 🔹 1. Filter noise
    data = filter_noise(data)

    # 🔹 2. Derived features
    systolic = data.get("systolic_bp", 120)
    diastolic = data.get("diastolic_bp", 80)

    data["pulse_pressure"] = systolic - diastolic

    # 🔹 3. Detect anomalies
    alerts = detect_anomalies(data)

    data["alerts"] = alerts

    return data