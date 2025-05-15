from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "nlp_app")

# Async client for FastAPI
async_client = AsyncIOMotorClient(MONGODB_URL)
db = async_client[DATABASE_NAME]

# Sync client for initial setup
sync_client = MongoClient(MONGODB_URL)
sync_db = sync_client[DATABASE_NAME]

# Create indexes
def create_indexes():
    # Create unique index on username
    sync_db.users.create_index("username", unique=True)

# Initialize database
def init_db():
    create_indexes()
    # Create admin user if it doesn't exist
    if not sync_db.users.find_one({"username": "admin"}):
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        sync_db.users.insert_one({
            "username": "admin",
            "hashed_password": pwd_context.hash("admin123"),
            "disabled": False
        }) 