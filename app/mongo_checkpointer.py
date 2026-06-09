"""
MongoEngine-backed LangGraph checkpointer.

Replaces MemorySaver with durable MongoDB storage.
All checkpoint blobs are serialised via LangGraph's built-in JsonPlusSerializer
(the same codec used by MemorySaver / SqliteSaver).
"""

import os
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import mongoengine as me
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


def _to_bytes(data: Any) -> Any:
    if isinstance(data, list):
        return bytes(data)
    return data


# ---------------------------------------------------------------------------
# MongoEngine Document model
# ---------------------------------------------------------------------------

class CheckpointDocument(me.Document):
    """Stores one LangGraph checkpoint snapshot per document."""

    thread_id            = me.StringField(required=True)
    checkpoint_id        = me.StringField(required=True)
    parent_checkpoint_id = me.StringField(default=None)

    # Full serialised checkpoint blob (channel values, versions, etc.)
    checkpoint           = me.DictField()

    # Step metadata (source, step number, writes, parents)
    metadata             = me.DictField()

    # Pending writes: list of {"task_id": ..., "channel": ..., "value": ...}
    pending_writes       = me.ListField(me.DictField())

    meta = {
        "collection": "checkpoints",
        "indexes": [
            # Fast lookup for get_tuple()
            {"fields": ["thread_id", "checkpoint_id"], "unique": True},
            # Fast ordered scan for list() — most recent first
            {"fields": ["thread_id", "-checkpoint_id"]},
        ],
    }


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

class MongoEngineCheckpointer(BaseCheckpointSaver):
    """
    LangGraph BaseCheckpointSaver backed by MongoEngine / MongoDB.

    Usage
    -----
    Call `MongoEngineCheckpointer.connect_db()` once before compiling the
    graph (e.g. at module import time in graph.py), then pass an instance
    as the `checkpointer` argument to `graph.compile()`.
    """

    serde = JsonPlusSerializer()

    # ------------------------------------------------------------------
    # DB connection (call once at app startup)
    # ------------------------------------------------------------------

    @classmethod
    def connect_db(cls, uri: Optional[str] = None, db: str = "skailama") -> None:
        """Connect MongoEngine to MongoDB.

        Parameters
        ----------
        uri : str, optional
            MongoDB connection string.  Falls back to ``MONGODB_URI`` env var.
        db : str
            Database name (default: ``"skailama"``).
        """
        connection_string = uri or os.getenv("MONGODB_URI")
        if not connection_string:
            raise RuntimeError(
                "MongoDB connection string not found. "
                "Set MONGODB_URI in your .env file."
            )
        me.connect(host=connection_string, db=db)
        logger.info("MongoEngine connected to MongoDB (db='%s')", db)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        """Serialize a Checkpoint object to a plain dict for storage."""
        return self.serde.dumps_typed(checkpoint)  # returns (type_str, bytes)

    def _deserialize_checkpoint(self, data: Dict[str, Any]) -> Checkpoint:
        """Deserialize stored dict back to a Checkpoint object."""
        return self.serde.loads_typed((data["type"], _to_bytes(data["data"])))

    def _serialize_metadata(self, metadata: CheckpointMetadata) -> Dict[str, Any]:
        return self.serde.dumps_typed(metadata)

    def _deserialize_metadata(self, data: Dict[str, Any]) -> CheckpointMetadata:
        return self.serde.loads_typed((data["type"], _to_bytes(data["data"])))

    @staticmethod
    def _serialize_writes(
        writes: Sequence[Tuple[str, str, Any]],
        serde: JsonPlusSerializer,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": task_id,
                "channel": channel,
                "type": type_str,
                "data": data_bytes,
            }
            for task_id, channel, (type_str, data_bytes) in (
                (t, c, serde.dumps_typed(v)) for t, c, v in writes
            )
        ]

    @staticmethod
    def _deserialize_writes(
        raw: List[Dict[str, Any]],
        serde: JsonPlusSerializer,
    ) -> List[Tuple[str, str, Any]]:
        return [
            (
                item["task_id"],
                item["channel"],
                serde.loads_typed((item["type"], _to_bytes(item["data"]))),
            )
            for item in raw
        ]

    def _doc_to_tuple(self, doc: CheckpointDocument) -> CheckpointTuple:
        checkpoint  = self._deserialize_checkpoint(doc.checkpoint)
        metadata    = self._deserialize_metadata(doc.metadata)
        writes      = self._deserialize_writes(doc.pending_writes or [], self.serde)
        config      = {
            "configurable": {
                "thread_id":       doc.thread_id,
                "checkpoint_id":   doc.checkpoint_id,
            }
        }
        parent_config = (
            {
                "configurable": {
                    "thread_id":     doc.thread_id,
                    "checkpoint_id": doc.parent_checkpoint_id,
                }
            }
            if doc.parent_checkpoint_id
            else None
        )
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=writes,
        )

    # ------------------------------------------------------------------
    # BaseCheckpointSaver interface
    # ------------------------------------------------------------------

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Return the latest (or a specific) checkpoint for a thread."""
        cfg           = config.get("configurable", {})
        thread_id     = cfg.get("thread_id")
        checkpoint_id = cfg.get("checkpoint_id")  # may be None → fetch latest

        if not thread_id:
            return None

        try:
            if checkpoint_id:
                doc = CheckpointDocument.objects(
                    thread_id=thread_id,
                    checkpoint_id=checkpoint_id,
                ).first()
            else:
                doc = (
                    CheckpointDocument.objects(thread_id=thread_id)
                    .order_by("-checkpoint_id")
                    .first()
                )
            return self._doc_to_tuple(doc) if doc else None
        except Exception as exc:
            logger.error("get_tuple failed: %s", exc, exc_info=True)
            return None

    def list(
        self,
        config: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoints for a thread, most recent first."""
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if not thread_id:
            return

        qs = CheckpointDocument.objects(thread_id=thread_id).order_by("-checkpoint_id")
        if before:
            before_id = before.get("configurable", {}).get("checkpoint_id")
            if before_id:
                qs = qs.filter(checkpoint_id__lt=before_id)
        if limit:
            qs = qs.limit(limit)

        for doc in qs:
            try:
                yield self._doc_to_tuple(doc)
            except Exception as exc:
                logger.warning("Skipping malformed checkpoint doc: %s", exc)

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Upsert a checkpoint snapshot to MongoDB."""
        cfg           = config.get("configurable", {})
        thread_id     = cfg["thread_id"]
        checkpoint_id = checkpoint["id"]
        parent_id     = cfg.get("checkpoint_id")  # previous checkpoint

        type_str_cp,  data_cp  = self.serde.dumps_typed(checkpoint)
        type_str_md,  data_md  = self.serde.dumps_typed(metadata)

        try:
            CheckpointDocument.objects(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
            ).update_one(
                set__parent_checkpoint_id=parent_id,
                set__checkpoint={"type": type_str_cp, "data": data_cp},
                set__metadata={"type": type_str_md, "data": data_md},
                set__pending_writes=[],
                upsert=True,
            )
        except Exception as exc:
            logger.error("put failed: %s", exc, exc_info=True)
            raise

        return {
            "configurable": {
                "thread_id":     thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, str, Any]],
        task_id: str,
    ) -> None:
        """Append pending writes to an existing checkpoint document."""
        cfg           = config.get("configurable", {})
        thread_id     = cfg["thread_id"]
        checkpoint_id = cfg["checkpoint_id"]

        serialized = self._serialize_writes(
            [(task_id, channel, value) for channel, value in writes],
            self.serde,
        )

        logger.info(f"[put_writes] thread_id={thread_id} checkpoint_id={checkpoint_id} task_id={task_id} writes_channels={[w[1] for w in writes]}")
        try:
            res = CheckpointDocument.objects(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
            ).update_one(push_all__pending_writes=serialized, upsert=True)
            logger.info(f"[put_writes] Update result: matched={res}")
        except Exception as exc:
            logger.error("put_writes failed: %s", exc, exc_info=True)
            raise
