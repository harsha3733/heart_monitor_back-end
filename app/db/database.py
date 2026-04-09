from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client = None
database = None  # renamed for clarity

async def connect_to_mongo():
    global client, database
    client = AsyncIOMotorClient(settings.MONGO_URI)
    database = client[settings.DB_NAME]
    print("✅ Connected to MongoDB")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("❌ MongoDB connection closed")

# ✅ ADD THIS FUNCTION
def get_db():
    return database