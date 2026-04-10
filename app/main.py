from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.database import connect_to_mongo, close_mongo_connection
from app.routes import auth_routes
from app.routes import profile_routes
from app.routes import sensor_routes
from app.routes import predict_routes

# ✅ Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    print("🚀 App started")

    yield

    # Shutdown
    await close_mongo_connection()
    print("🛑 App stopped")


app = FastAPI(
    title="Heart Disease Prediction API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_routes.router)
app.include_router(profile_routes.router)
app.include_router(sensor_routes.router)
app.include_router(predict_routes.router)

@app.get("/")
def root():
    return {"message": "API is running"}