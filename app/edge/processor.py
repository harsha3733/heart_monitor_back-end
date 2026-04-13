from app.edge.filters import filter_noise
from app.edge.anomaly import detect_anomalies


def process_sensor_data(data: dict):
    """
    Edge pipeline:
    - Filter noise / clamp to physiological ranges
    - Anomaly detection
    Note: pulse_pressure is derived from PROFILE (systolic_bp - diastolic_bp)
    inside predict_service, not here — sensor data doesn't carry BP readings.
    """

    # 1. Filter noise
    data = filter_noise(data)

    # 2. Detect anomalies
    alerts = detect_anomalies(data)
    data["alerts"] = alerts

    return data
