def detect_anomalies(data: dict):
    """
    Detect critical health conditions
    """

    alerts = []

    if data["heart_rate"] > 120:
        alerts.append("High Heart Rate")

    if data["heart_rate"] < 50:
        alerts.append("Low Heart Rate")

    if data["spo2"] < 90:
        alerts.append("Low Oxygen Level")

    if data["temperature"] > 38:
        alerts.append("Fever Detected")

    return alerts