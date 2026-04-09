def filter_noise(data: dict):
    """
    Clean sensor noise and normalize values
    """

    data["heart_rate"] = max(40, min(data.get("heart_rate", 75), 180))
    data["spo2"] = max(80, min(data.get("spo2", 98), 100))
    data["temperature"] = round(data.get("temperature", 36.5), 1)

    return data