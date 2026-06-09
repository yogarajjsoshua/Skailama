import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "skailama")
MONGODB_COLLECTION_CHATS = os.getenv("MONGODB_COLLECTION_CHATS", "chats")
MONGODB_COLLECTION_MESSAGES = os.getenv("MONGODB_COLLECTION_MESSAGES", "messages")
MONGODB_COLLECTION_LICENSES = os.getenv("MONGODB_COLLECTION_LICENSES", "licenses")
