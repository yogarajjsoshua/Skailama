import logging
from datetime import datetime
from typing import Literal, List, Dict, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from ..config import (
    MONGODB_URI,
    MONGODB_DB_NAME,
    MONGODB_COLLECTION_CHATS,
    MONGODB_COLLECTION_MESSAGES,
    MONGODB_COLLECTION_LICENSES,
)

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB_NAME]
        self.development_db = self.client["Development"]
        self.chats: Collection = self.db[MONGODB_COLLECTION_CHATS]
        self.messages: Collection = self.db[MONGODB_COLLECTION_MESSAGES]
        self.licenses: Collection = self.development_db[MONGODB_COLLECTION_LICENSES]
        logger.info(
            "Connected to MongoDB at %s, db=%s, collections=(%s,%s)",
            MONGODB_URI,
            MONGODB_DB_NAME,
            MONGODB_COLLECTION_CHATS,
            MONGODB_COLLECTION_MESSAGES,
        )

    def get_license_id(self, user_id: str) -> str:
        doc = self.licenses.find_one({"userID": user_id}, {"_id": 0, "licenseKey": 1})
        return doc.get("licenseKey", "") if doc else ""

    def create_chat_if_missing(self, user_id: str, chat_id: str):
        now = datetime.now().isoformat()
        self.chats.update_one(
            {"user_id": user_id, "chat_id": chat_id},
            {
                "$setOnInsert": {
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "created_at": now,
                    "last_context": {},
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
        )
        logger.debug("Ensured chat exists user_id=%s chat_id=%s", user_id, chat_id)
        return (user_id, chat_id, now, now)

    def save_chat_context(self, user_id: str, chat_id: str, context: dict):
        now = datetime.now().isoformat()
        self.chats.update_one(
            {"user_id": user_id, "chat_id": chat_id},
            {"$set": {"last_context": context, "updated_at": now}},
            upsert=True,
        )
        logger.debug(
            "Saved chat context user_id=%s chat_id=%s node=%s",
            user_id,
            chat_id,
            context.get("current_node"),
        )

    def get_chat_context(self, user_id: str, chat_id: str) -> dict:
        doc = self.chats.find_one({"user_id": user_id, "chat_id": chat_id}, {"_id": 0, "last_context": 1})
        ctx = doc.get("last_context", {}) if doc else {}
        logger.debug("Fetched chat context user_id=%s chat_id=%s has_context=%s", user_id, chat_id, bool(ctx))
        return ctx

    def save_message(
        self,
        user_id: str,
        chat_id: str,
        message_id: str,
        role: Literal["user", "assistant"],
        content: str,
        timestamp: str,
        ui: dict = None,
    ):
        doc = {
            "user_id": user_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }
        if ui:
            doc["ui"] = ui
        self.messages.insert_one(doc)
        logger.debug(
            "Saved message user_id=%s chat_id=%s role=%s message_id=%s has_ui=%s",
            user_id,
            chat_id,
            role,
            message_id,
            bool(ui),
        )

    def list_chats(self, user_id: str) -> List[Dict[str, Any]]:
        return [
            {k: d[k] for k in ["user_id", "chat_id", "created_at", "updated_at"] if k in d}
            for d in self.chats.find({"user_id": user_id}, {"_id": 0}).sort("updated_at", -1)
        ]

    def get_messages(self, user_id: str, chat_id: str) -> List[Dict[str, Any]]:
        return [
            {k: d[k] for k in ["message_id", "role", "content", "timestamp", "ui"] if k in d}
            for d in self.messages.find({"user_id": user_id, "chat_id": chat_id}, {"_id": 0}).sort("timestamp", 1)
        ]

__all__ = ["MongoDBClient", "PyMongoError"]
