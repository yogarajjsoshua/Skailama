#!/usr/bin/env python
import os
import logging
from dotenv import load_dotenv
import mongoengine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("check_mongo")

def main():
    logger.info("Loading environment variables...")
    load_dotenv()
    
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("MONGODB_URI is not set in the environment or .env file.")
        return
        
    logger.info("Connecting to MongoDB...")
    try:
        # Initialize connection using mongoengine (like the checkpointer does)
        mongoengine.connect(host=mongodb_uri)
        
        # Access underlying pymongo client and run ping command
        from mongoengine.connection import get_connection
        client = get_connection()
        
        logger.info("Sending ping command to MongoDB...")
        client.admin.command('ping')
        
        logger.info("MongoDB connection verification: SUCCESS ✓")
    except Exception as e:
        logger.error(f"MongoDB connection verification: FAILED ✗\nError: {e}")

if __name__ == "__main__":
    main()
