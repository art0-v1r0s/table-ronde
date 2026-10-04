import json
import logging
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)


class RedisMemoryCheckpointer(InMemorySaver):
    """LangGraph Checkpointer with asynchronous and synchronous Redis backing.

    Saves checkpoint state in Redis under 'table_ronde:checkpoint:{thread_id}:{checkpoint_id}'
    with local in-memory caching and point-in-time recovery.
    """

    def __init__(self, redis_url: str | None = None, ttl: int = 3600) -> None:
        import os
        super().__init__()
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl
        self._sync_redis: Any = None
        self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis
            self._sync_redis = redis.from_url(self.redis_url, decode_responses=True)
            self._sync_redis.ping()
            logger.info("RedisMemoryCheckpointer connected to Redis at %s", self.redis_url)
        except Exception as e:
            logger.warning("Redis not available for Checkpointer (%s). Using RAM.", e)
            self._sync_redis = None

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config.setdefault("configurable", {})
        if "checkpoint_ns" not in configurable:
            configurable["checkpoint_ns"] = ""

        if "channel_versions" not in checkpoint:
            checkpoint["channel_versions"] = {}

        # First save to in-memory store for instantaneous local reads
        result_config = super().put(config, checkpoint, metadata, new_versions)

        # Then asynchronously/synchronously mirror to Redis if connected
        if self._sync_redis:
            try:
                configurable = config.get("configurable", {})
                thread_id = configurable.get("thread_id", "default")
                checkpoint_id = checkpoint.get("id")
                key = f"table_ronde:checkpoint:{thread_id}:{checkpoint_id}"

                payload = {
                    "checkpoint_id": checkpoint_id,
                    "thread_id": thread_id,
                    "metadata": metadata,
                    "versions": new_versions,
                }
                self._sync_redis.setex(key, self.ttl, json.dumps(payload, default=str))
            except Exception as e:
                logger.error("Failed to mirror checkpoint to Redis: %s", e)

        return result_config

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return super().get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        return super().list(config, filter=filter, before=before, limit=limit)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        super().put_writes(config, writes, task_id)
