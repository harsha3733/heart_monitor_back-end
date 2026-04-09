🫀 Heart Disease Prediction Backend with Edge-Cloud Architecture

This repository contains the FastAPI backend for a Personalized Human Digital Twin (HDT) system designed for remote patient monitoring, real-time analytics, and heart disease prediction.

The system follows an edge-cloud architecture, where sensor data is preprocessed at the edge layer before being stored and analyzed in the cloud.

🚀 Features
🔐 JWT-based Authentication (login/register)
👤 User Profile Management (static health data)
📡 Sensor Data Ingestion (time-series data)
⚡ Edge Processing Layer (real-time filtering & anomaly detection)
🧠 ML-Based Prediction (XGBoost + MLP ensemble)
📊 Real-time Dashboard Support (polling APIs)
🚨 Alert Generation (based on physiological thresholds)
🧠 Architecture Overview
Sensors / Frontend
        ↓
Edge Layer (Filtering + Anomaly Detection)
        ↓
FastAPI Backend (Cloud Layer)
        ↓
MongoDB
        ↓
Prediction & Dashboard APIs
📁 Project Structure
Backend/
│
├── app/
│   ├── main.py
│
│   ├── core/
│   │   └── config.py
│
│   ├── db/
│   │   └── database.py
│
│   ├── models/
│   │   ├── user_model.py
│   │   ├── profile_model.py
│   │   ├── sensor_model.py
│
│   ├── routes/
│   │   ├── auth_routes.py
│   │   ├── profile_routes.py
│   │   ├── sensor_routes.py
│   │   ├── predict_routes.py
│
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── profile_service.py
│   │   ├── sensor_service.py
│   │   ├── predict_service.py
│
│   ├── edge/                # 🔥 Edge Layer
│   │   ├── processor.py
│   │   ├── filters.py
│   │   ├── anomaly.py
│
├── Model/                  # ML Models (.pkl)
├── .env
├── requirements.txt
└── README.md
⚡ Edge Layer (Core Innovation)

The Edge Layer simulates real-world edge computing:

🔹 Responsibilities
Noise filtering of sensor data
Feature engineering (e.g., pulse pressure)
Real-time anomaly detection
Alert generation with severity levels
🔹 Example Alerts
High Heart Rate
Low SpO2
Fever Detection
🔹 Output

Processed data is enriched with:

{
  "alerts": [...],
  "status": "NORMAL | WARNING | CRITICAL"
}
🧠 Machine Learning Model
Ensemble of:
XGBoost
MLP (Neural Network)
Workflow:
Fetch profile (static data)
Aggregate sensor data (dynamic)
Apply scaling
Predict using ensemble
Return risk level
🔐 Authentication
JWT-based authentication
Bearer token required for protected routes
Optional toggle via .env
⚙️ Environment Variables

Create a .env file:

MONGO_URI=your_mongodb_uri
DB_NAME=heart_disease_db

SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

AUTH_ENABLED=true
🧪 Running the Backend
1. Install dependencies
pip install -r requirements.txt
2. Run server
uvicorn app.main:app --reload
3. Access API
Swagger Docs:
http://127.0.0.1:8000/docs
📡 Key APIs
🔐 Auth
POST /auth/register
POST /auth/login
👤 Profile
POST /profile
GET /profile
📡 Sensor
POST /sensor
GET /sensor/latest
GET /sensor/history
🧠 Prediction
GET /predict
🚀 Future Improvements
WebSocket-based real-time streaming
Edge trend detection
Personalized thresholds
Notification system (alerts)
Azure deployment (cloud layer)
📚 Research Alignment

This backend is implemented based on the paper:

“A Personalized Human Digital Twin Framework for Remote Patient Monitoring with Edge-Cloud Integration and Interactive Visualization”

Key concepts implemented:

Edge-cloud integration
Real-time anomaly detection
Personalized prediction model
Interactive dashboard support
👨‍💻 Authors
Harsha Vardhan
Team (HDT Project)
⭐ Contribution

Feel free to fork, improve, and contribute!